#!/usr/bin/env python3
"""Run one MEMORIUM U05 arm over a set of memory units (conversations/scenarios) in order.

Units are processed whole and in question order, because write-back makes answers depend on the
questions that came before. Sharding is therefore by unit, never by question index.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from memorium import SourceRecord, TaskSpec
from memorium.core import View
from memorium.u05 import ArmConfig, build_indices, run_task
from memorium.unit import UnitConfig, build_unit
from memorium.views import MaterializationPolicy, ViewConfig, node_kind

ARMS = {
    "A0": dict(write_back=False, t1=False, v2=False, structure=False, reranker=False, adapter=False),
    "F_v2": dict(write_back=False, t1=False, v2=True, structure=True, reranker=True, adapter=True),
    "F_v2_zero": dict(write_back=False, t1=False, v2=True, structure=True, reranker=True, adapter=False),
    "F_lite": dict(write_back=False, t1=False, v2=False, structure=True, reranker=True, adapter=True),
    "F_lite_zero": dict(write_back=False, t1=False, v2=False, structure=True, reranker=True, adapter=False),
    "A0_lora": dict(write_back=False, t1=False, v2=False, structure=False, reranker=False, adapter=True),
}


def pointers_of(result) -> set[str]:
    """Experience nodes this task's reader saw as source pointers only (admission decision "pointer")."""
    log = getattr(result, "experience_admission", None) or {}
    decision = log.get("admission") or {}
    return {decision["node"]} if decision.get("decision") == "pointer" else set()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def redirect_views(materializer, directory: Path) -> None:
    """Send answer-time renders and V2 images to this run's directory, keeping the shared build cache
    read-only. Lives here, not in memorium/materialize.py, because build code is part of the cache key.
    File names are content-addressed, so results do not change."""
    from memorium.adapters.render import PageRenderer

    materializer.directory = Path(directory)
    materializer.directory.mkdir(parents=True, exist_ok=True)
    materializer.log = materializer.directory / "materialize.jsonl"
    if materializer._renderer is not None:
        materializer._renderer = PageRenderer(materializer.directory / "pages")
    if materializer._imagegen is not None:
        materializer._imagegen.directory = materializer.directory / "v2"
        materializer._imagegen.directory.mkdir(parents=True, exist_ok=True)


def attach_session_pages(unit, materializer, calls, wrap: int) -> dict:
    """Render each session's full text as page views on the session node (long documents: one chunk per
    session). Outside build code so caches stay valid; pages are content-addressed. Returns counts."""
    import textwrap
    from dataclasses import replace

    from memorium.core import MemoryPatch, View

    upserts, pages, tokens = [], 0, 0
    for session_id in unit.sessions:
        node = unit.memory.nodes[session_id]
        if any(v.kind == "V1" for v in node.views):
            continue
        text = "\n".join(unit.memory.sources[ref].text for ref in node.source_refs)
        lines = [piece for line in text.splitlines() for piece in (textwrap.wrap(line, wrap) or [])]
        if not lines:
            continue
        try:
            rendered = materializer.renderer.render(session_id, lines)
        except ValueError as exc:
            if "unsupported glyphs" not in str(exc):
                raise
            continue
        views = []
        for index, (path, _) in enumerate(rendered):
            cost = calls.count("", (path,))
            views.append(View("V1", path, node.source_refs, "canvas_renderer_session", cost, True,
                              {"page": index, "pages": len(rendered), "wrap": wrap}))
            pages += 1
            tokens += cost
        upserts.append(replace(node, views=node.views + tuple(views), version=0, source_citations=(), content_hash=""))
    if upserts:
        unit.memory.apply(MemoryPatch(f"{unit.unit_id}:session_pages:{unit.memory.revision + 1}",
                                      expected_revision=unit.memory.revision, upsert=tuple(upserts)))
    return {"pages": pages, "vision_tokens": tokens}


def tolerate_missing_glyphs(materializer, log: list) -> None:
    """A rendered page whose text the font cannot draw is skipped and recorded, never substituted.

    The node keeps its text views. Lives in the runner, not in build code, so existing caches stay valid.
    """
    render_v1 = materializer.v1

    def v1(unit, node_id):
        try:
            return render_v1(unit, node_id)
        except ValueError as exc:
            if "unsupported glyphs" not in str(exc):
                raise
            log.append({"node": node_id, "reason": str(exc)[:200]})
            return None

    materializer.v1 = v1


def main() -> int:
    model_root = Path(os.environ.get("MMMEM_WORKSPACE", Path.home() / "mmmem-workspace")) / "models"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True, help="split JSON with train/test unit ids")
    parser.add_argument("--part", choices=("test", "train"), default="test")
    parser.add_argument("--arm", choices=sorted(ARMS), required=True)
    parser.add_argument("--routes", default="TV", help="T, V or TV")
    parser.add_argument("--reader-memory", choices=("text", "image", "dense"), default="text",
                        help="dense: select exactly as text, then show the selected text as dense pages")
    parser.add_argument("--page-font", type=int, default=8, help="font size for --reader-memory dense")
    parser.add_argument("--iterative", action="store_true")
    parser.add_argument("--v2-coverage", type=float, default=1.0)
    parser.add_argument("--token-budget", type=int, default=3000)
    parser.add_argument("--note-pull", choices=("segments", "none"), default="segments")
    parser.add_argument("--note-limit", type=int, default=None)
    parser.add_argument("--note-min-sim", type=float, default=None)
    parser.add_argument("--note-style", choices=("answer", "evidence"), default="answer")
    parser.add_argument("--units", default=None, help="comma-separated subset of unit ids (sharding)")
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reader-model", default=str(model_root / 'Qwen3.5-9B'))
    parser.add_argument("--reader-backend", default="qwen35")
    parser.add_argument("--reader-adapter", default=None)
    parser.add_argument("--reranker", default=None)
    parser.add_argument("--preserve-query-image-rank", action="store_true",
                        help="fuse picture-route and text-reranker ranks; default off")
    parser.add_argument("--read-generated-views", action="store_true",
                        help="send a selected node's generated scene view (V2) to the reader with its "
                             "text, inside the token budget; default off (V2 then only feeds the visual index)")
    parser.add_argument("--text-encoder", default=str(model_root / 'Qwen3-Embedding-0.6B'))
    parser.add_argument("--visual-encoder", default=str(model_root / 'Qwen3-VL-Embedding-2B'))
    parser.add_argument("--image-model", default=str(model_root / 'Qwen-Image'))
    parser.add_argument("--prompt-mode", choices=("memorium", "loogle"), default="memorium",
                        help="loogle: the official LooGLE-v2 instruction template with answer-first suffix")
    parser.add_argument("--v1-level", choices=("segment", "session", "off"), default="segment",
                        help="where rendered pages live: per segment (default), per session (long documents), or none")
    parser.add_argument("--page-wrap", type=int, default=100, help="characters per rendered line for session pages")
    parser.add_argument("--native-max-side", type=int, default=640,
                        help="downscale native photos to this maximum side before use (V0 views)")
    parser.add_argument("--build-cache", type=Path, default=None,
                        help="directory of content-addressed unit builds shared across arms")
    parser.add_argument("--build-first", action="store_true",
                        help="build every unit's views before answering any question (units are independent)")
    parser.add_argument("--wait-for-adapter", type=int, default=0,
                        help="seconds to wait for --reader-adapter to appear after building; 0 = must exist now")
    parser.add_argument("--v2-unit-budget", type=int, default=48, help="generated integration images per unit")
    parser.add_argument("--experience", choices=("off", "on"), default="off",
                        help="write an experience node after every task; staged-first route; adoption-based promotion")
    parser.add_argument("--exp-image", choices=("off", "stage", "adopt", "promote"), default="adopt")
    parser.add_argument("--exp-image-budget", type=int, default=24)
    parser.add_argument("--exp-no-summary", action="store_true")
    parser.add_argument("--exp-no-promotion", action="store_true")
    parser.add_argument("--exp-no-staging", action="store_true")
    parser.add_argument("--exp-promote-uses", type=int, default=3)
    parser.add_argument("--exp-promote-families", type=int, default=2, help="question families a node must be adopted across")
    parser.add_argument("--prereg", type=Path, default=None, help="frozen protocol file; its SHA-256 is recorded in the manifest")
    parser.add_argument("--exp-topk", type=int, default=3)
    parser.add_argument("--exp-supported-only", action="store_true")
    parser.add_argument("--warmup-tasks", type=Path, default=None,
                        help="rsi.jsonl written by run_memorium_joint_rsi.py --warmup: played before each unit's "
                             "scored questions (experience nodes are built, nothing is scored)")
    parser.add_argument("--exp-outside", action="store_true",
                        help="top-1 experience node is added after evidence selection, outside the token budget")
    parser.add_argument("--exp-admission", choices=("off", "on"), default="off",
                        help="picture-scoped pre-reader admission of the top-1 outside node (full/pointer/skip; "
                             "memorium/experience.py); off = historical behaviour; needs --exp-outside")
    parser.add_argument("--abstain-edit", action="store_true",
                        help="on abstention, re-read once with a task-edited evidence sheet from a wider pool")
    parser.add_argument("--task-order", choices=("dataset", "reverse"), default="dataset",
                        help="question stream order within a unit (staging is order-dependent)")
    parser.add_argument("--dry-run", action="store_true", help="hash encoders, echo reader; never reportable")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("--output must not already exist")
    arm_spec = ARMS[args.arm]
    if arm_spec["reranker"] and not (args.reranker or args.dry_run):
        parser.error(f"{args.arm} requires --reranker")
    if arm_spec["adapter"] and not (args.reader_adapter or args.dry_run):
        parser.error(f"{args.arm} requires --reader-adapter")
    if args.wait_for_adapter and not (args.reader_adapter and args.build_first):
        parser.error("--wait-for-adapter needs --reader-adapter and --build-first")
    # Never let a model flag be silently ignored by an arm that does not use it.
    if args.reranker and not arm_spec["reranker"]:
        parser.error(f"{args.arm} does not use a reranker; --reranker would be ignored")
    if args.reader_adapter and not arm_spec["adapter"]:
        parser.error(f"{args.arm} does not use a reader adapter; --reader-adapter would be ignored")
    cell, output = args.cell.resolve(), args.output.resolve()
    split = json.loads(args.split.read_text(encoding="utf-8"))
    unit_ids = list(split[args.part])
    if args.units:
        wanted = args.units.split(",")
        if set(wanted) - set(unit_ids):
            parser.error("--units must be a subset of the chosen split part")
        unit_ids = [unit for unit in unit_ids if unit in wanted]

    sources_by_unit: dict[str, list[SourceRecord]] = defaultdict(list)
    for row in read_jsonl(cell / "source_records.jsonl"):
        unit = row["locator"].get("sample_id") or row["locator"].get("scenario")
        if unit in unit_ids:
            sources_by_unit[unit].append(SourceRecord(row["id"], row["ordinal"], row["text"], row["locator"],
                                                      tuple(row.get("images", ()))))
    tasks_by_unit: dict[str, list[dict]] = defaultdict(list)
    for row in read_jsonl(cell / "public_tasks.jsonl"):
        if row["sample_id"] in unit_ids:
            tasks_by_unit[row["sample_id"]].append(row)

    warm_rows = []
    if args.warmup_tasks:
        if args.experience != "on":
            parser.error("--warmup-tasks needs --experience on")
        warm_rows = read_jsonl(args.warmup_tasks)
    if args.exp_admission == "on" and (args.experience != "on" or not args.exp_outside):
        parser.error("--exp-admission on needs --experience on and --exp-outside")
    view_config = ViewConfig(t1=arm_spec["t1"], v1=args.v1_level == "segment", v2=arm_spec["v2"],
                             v2_coverage=args.v2_coverage, v2_unit_budget=args.v2_unit_budget)
    arm = ArmConfig(routes=tuple(args.routes), write_back=arm_spec["write_back"], iterative=args.iterative,
                    reader_memory=args.reader_memory, token_budget=args.token_budget,
                    note_pull=args.note_pull, note_limit=args.note_limit, note_min_sim=args.note_min_sim,
                    note_style=args.note_style,
                    preserve_query_image_rank=args.preserve_query_image_rank,
                    read_generated_views=args.read_generated_views)
    output.mkdir(parents=True)
    manifest = {
        "schema": "memorium-u05-cell-run-v1", "status": "STARTED", "arm": args.arm,
        "arm_components": arm_spec, "routes": args.routes, "reader_memory": args.reader_memory,
        "iterative": args.iterative, "v2_coverage": args.v2_coverage, "token_budget": args.token_budget,
        "page_font": args.page_font if args.reader_memory == "dense" else None,
        "write_back_controls": {"note_pull": args.note_pull, "note_limit": args.note_limit,
                                "note_min_sim": args.note_min_sim, "note_style": args.note_style},
        "part": args.part, "units": unit_ids, "split_sha256": sha256_file(args.split),
        "input_sha256": {name: sha256_file(cell / name) for name in ("source_records.jsonl", "public_tasks.jsonl")},
        "code_sha256": {str(p.resolve().relative_to(ROOT)): sha256_file(p) for p in (
            Path(__file__), ROOT / "memorium/u05.py", ROOT / "memorium/unit.py", ROOT / "memorium/core.py",
            ROOT / "memorium/structure.py", ROOT / "memorium/views.py")},
        "encoders": "hash_dryrun_not_reportable" if args.dry_run else {
            "text": args.text_encoder, "visual": args.visual_encoder},
        "component_mode": {"analyzer": "public_query", "entities": "deterministic_capitalised_names",
                           "timeline": "deterministic_relative_time_rules",
                           "selector": ("zero_shot_reranker" if args.arm in ("A5_zero", "A5_lora") else "trained_reranker")
                           if arm_spec["reranker"] else "rrf_tv",
                           "reader": "lora" if arm_spec["adapter"] else "frozen"},
        "host": socket.gethostname(), "started_at": datetime.now(timezone.utc).isoformat(),
        "dry_run": args.dry_run, "build_first": args.build_first,
        "build_cache_root": str(args.build_cache) if args.build_cache else None,
        "prompt_mode": args.prompt_mode, "v1_level": args.v1_level, "page_wrap": args.page_wrap,
        "reader_adapter": args.reader_adapter, "reranker": args.reranker,
        **({"preserve_query_image_rank": True} if args.preserve_query_image_rank else {}),
        **({"read_generated_views": True} if args.read_generated_views else {}),
        "task_order": args.task_order,
        "experience_settings": {
            "enabled": args.experience == "on", "image": args.exp_image, "image_budget": args.exp_image_budget,
            "summary": not args.exp_no_summary, "promotion": not args.exp_no_promotion,
            "staging": not args.exp_no_staging, "promote_uses": args.exp_promote_uses,
            "promote_families": args.exp_promote_families, "top_k": args.exp_topk,
            "supported_only": args.exp_supported_only, "outside": args.exp_outside,
            "admission": args.exp_admission, "abstain_edit": args.abstain_edit,
        },
    }
    write_json(output / "run_manifest.json", manifest)

    current_query_image = [None]
    current_task = [None]
    # Cells built before manifests existed (e.g. the LoCoMo option probe) carry no system prompt.
    manifest_path = cell / "cell_manifest.json"
    if args.prereg:
        manifest["prereg"] = {"path": str(args.prereg), "sha256": hashlib.sha256(args.prereg.read_bytes()).hexdigest()}
    if warm_rows:
        manifest["warmup"] = {"path": str(args.warmup_tasks), "rows": len(warm_rows),
                              "sha256": hashlib.sha256(args.warmup_tasks.read_bytes()).hexdigest()}
    cell_manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    system_prompt = cell_manifest.get("system_prompt", "")
    from memorium.adapters.route_index import HashEncoder, RouteIndex
    if args.dry_run:
        text_index, visual_index = RouteIndex(HashEncoder()), RouteIndex(HashEncoder()) if "V" in args.routes else None

        def reader(prompt, views):
            first = views[0].content.splitlines()[0] if views else ""
            return {"output": f"Answer: {first[:60]}\nEvidence:", "input_tokens": 0, "input_vision_tokens": 0,
                    "status": "OK"}
    else:
        from memorium.adapters.route_index import QwenTextEncoder, QwenVisualEncoder
        from memorium.readers import make_calls
        text_index = RouteIndex(QwenTextEncoder(args.text_encoder))
        visual_index = RouteIndex(QwenVisualEncoder(args.visual_encoder)) if "V" in args.routes else None
        # Views (T1, V2 specs and checks) always come from the frozen model: an adapter-bearing ModelCalls
        # refuses every stage but "answerer", and views must not depend on the reader under test.
        calls = make_calls(args.reader_model, args.reader_backend, output / "calls", max_input_tokens=32768)
        answer_calls = [None if args.reader_adapter else calls]

        dense_pages = None
        if args.reader_memory == "dense":
            from memorium.adapters.dense_page import DensePageRenderer
            dense_pages = DensePageRenderer(output / "dense_pages", font_size=args.page_font)

        def reader(prompt, views):
            texts = [v.content for v in views if v.modality == "text"]
            # Generated scene views (only present with --read-generated-views) go last and are named below.
            generated = [v.content for v in views if v.kind == "V2"]
            images = [v.content for v in views if v.modality == "image" and v.kind != "V2"]
            if dense_pages is not None and texts:
                # Same evidence, same order as the text arm; only its form changes.
                lines = "\n\n".join(texts).splitlines()
                key = hashlib.sha256("\n".join(lines).encode()).hexdigest()[:24]
                images = [path for path, _ in dense_pages.render(key, lines)] + images
                texts = ["[The retrieved text is attached as page images.]"]
            body = "\n\n".join(texts)
            if args.prompt_mode == "loogle":
                from memorium.adapters.loogle import GENERIC_ANSWER_FIRST_SUFFIX, official_prompt
                raw = current_task[0]
                context = body if body.strip() else ""
                images = images + generated
                if images:
                    context = (context + "\n\n" if context else "") + "[Further retrieved pages are attached as images.]"
                item = {"instruction": raw["instruction"], "question": raw["loogle_question"],
                        "options": raw["loogle_options"]}
                return answer_calls[0].call("answerer", official_prompt(item, context) + GENERIC_ANSWER_FIRST_SUFFIX,
                                            images=images, max_new_tokens=128)
            if current_query_image[0]:
                # The question's own picture comes first and is named so the reader can tell it apart.
                images = [current_query_image[0]] + images
                prompt = prompt + "\n(The first image is the picture shown with the question.)"
            if generated:
                images = images + generated
                prompt += f"\n(The last {len(generated)} image(s) are generated illustrations of remembered text.)"
            if system_prompt:
                prompt = f"{system_prompt}\n\n{prompt}"
            return answer_calls[0].call("answerer", f"{prompt}\n\nMemory:\n{body}", images=images,
                                        max_new_tokens=128)

    materializer = None
    if not args.dry_run:
        from memorium.materialize import Materializer
        materializer = Materializer(output / "views", reader_calls=calls if not args.dry_run else None,
                                    image_model=args.image_model if view_config.v2 else None)
        manifest["v1_skipped_missing_glyphs"] = []
        tolerate_missing_glyphs(materializer, manifest["v1_skipped_missing_glyphs"])
        manifest["canvas_font"] = os.environ.get("VMC_CANVAS_FONT")

    reranker = None
    if arm_spec["reranker"] and not args.dry_run:
        from memorium.training.reranker import load_reranker
        reranker = load_reranker(args.reranker)

    if args.exp_admission == "on":
        manifest["code_sha256"]["memorium/experience.py"] = sha256_file(ROOT / "memorium/experience.py")

    exp_draw = None
    if args.experience == "on" and args.exp_image != "off" and not args.dry_run:
        from memorium.adapters.imagegen import QwenImageGenerator
        exp_gen = QwenImageGenerator(args.image_model, output / "experience_images", size=512, steps=20)
        exp_draw = lambda scene, seed: exp_gen(scene, seed)

    policy = MaterializationPolicy(view_config)
    rows_path = output / "rows.jsonl"
    started = time.perf_counter()
    total = 0
    unit_config = lambda: UnitConfig(views=view_config, structure=arm_spec["structure"])
    cache = None
    if args.build_cache is not None:
        from memorium.cache import BuildCache, build_key
        key, spec = build_key(sources_sha=manifest["input_sha256"]["source_records.jsonl"],
                              unit_config=unit_config(), structure=arm_spec["structure"],
                              writer=None if args.dry_run else f"{args.reader_model}|{args.reader_backend}",
                              image_model=args.image_model, font=os.environ.get("VMC_CANVAS_FONT"),
                              dry_run=args.dry_run)
        cache = BuildCache(args.build_cache, key, spec)
        manifest["build_cache"] = {"dir": str(cache.dir), "key": key, "units": {}}
        if materializer is not None:
            # Build-time images and logs live with the cache; answer-time upgrades stay with this run.
            materializer.directory = cache.views_dir
            materializer.directory.mkdir(parents=True, exist_ok=True)

    def fresh(unit_id):
        unit = build_unit(unit_id, sources_by_unit[unit_id], unit_config())
        if materializer is not None:
            materializer.log = (cache.views_dir if cache else output / "views") / f"build_{unit_id}.jsonl"
            materializer.build_views(unit, policy)
        return unit

    def build(unit_id):
        if cache is None:
            unit = fresh(unit_id)
        else:
            unit, how = cache.get_or_build(unit_id, sources_by_unit[unit_id], unit_config(),
                                           lambda: fresh(unit_id))
            manifest["build_cache"]["units"][unit_id] = {"source": how, "snapshot_hash": unit.memory.snapshot_hash}
            write_json(output / "run_manifest.json", manifest)
        if materializer is not None:
            redirect_views(materializer, output / "views")
        if args.v1_level == "session" and materializer is not None:
            manifest.setdefault("session_pages", {})[unit_id] = attach_session_pages(
                unit, materializer, calls, args.page_wrap)
        if any(unit.memory.sources[ref].images for ref in unit.memory.sources):
            from memorium.native import attach_native_views
            count = (lambda path: 256) if args.dry_run else (lambda path: calls.count("", (path,)))
            manifest.setdefault("native_views", {})[unit_id] = attach_native_views(
                unit, directory=output / "native", count_image=count, max_side=args.native_max_side)
        build_indices(unit, text_index, visual_index)
        return unit

    def ensure_answerer():
        if args.dry_run or answer_calls[0] is not None:
            return
        adapter = Path(args.reader_adapter)
        deadline = time.time() + args.wait_for_adapter
        while not (adapter / "adapter_model.safetensors").is_file():
            if time.time() > deadline:
                raise FileNotFoundError(f"reader adapter not ready: {adapter}")
            time.sleep(60)
        time.sleep(30)  # let the trainer finish writing
        # One backbone: the adapter is on for "answerer" and off for view writing (see shared_reader).
        from memorium.adapters.shared_reader import SharedCalls
        SharedCalls(calls, str(adapter))
        answer_calls[0] = calls
        manifest["reader_adapter_loaded_at"] = datetime.now(timezone.utc).isoformat()
        write_json(output / "run_manifest.json", manifest)

    prebuilt = {}
    if args.build_first:
        for unit_id in unit_ids:
            unit_started = time.perf_counter()
            prebuilt[unit_id] = (build(unit_id), time.perf_counter() - unit_started)
    ensure_answerer()
    with rows_path.open("w", encoding="utf-8") as rows, (output / "warmup_rows.jsonl").open("w", encoding="utf-8") as warm_out:
        for unit_id in unit_ids:
            unit_started = time.perf_counter()
            if unit_id in prebuilt:
                unit, build_seconds = prebuilt.pop(unit_id)
                unit_started -= build_seconds
            else:
                unit = build(unit_id)
            tasks = sorted(tasks_by_unit[unit_id], key=lambda row: row["question_index"],
                           reverse=args.task_order == "reverse")
            if args.max_tasks is not None:
                tasks = tasks[:args.max_tasks]
            if warm_rows:
                from memorium.experience import warmup_tasks
                tasks = warmup_tasks(warm_rows, unit_id) + tasks
            exp = None
            if args.experience == "on":
                from memorium.experience import ExperienceStore, Settings as ExpSettings
                exp = ExperienceStore(
                    unit, RouteIndex(text_index.encoder),
                    ExpSettings(image=args.exp_image, image_budget=args.exp_image_budget,
                                summary=not args.exp_no_summary, promotion=not args.exp_no_promotion,
                                staging=not args.exp_no_staging, promote_uses=args.exp_promote_uses, promote_families=args.exp_promote_families,
                                top_k=args.exp_topk, supported_only=args.exp_supported_only, outside=args.exp_outside,
                                admission=args.exp_admission == "on"),
                    text_index=text_index, visual_index=visual_index,
                    summarize=None if args.dry_run else (
                        lambda prompt: str(answer_calls[0].call("answerer", prompt, max_new_tokens=96).get("output", ""))),
                    draw=exp_draw, count_image=None if args.dry_run else (lambda path: calls.count("", (path,))))
            for step, raw_task in enumerate(tasks):
                current_query_image[0] = raw_task.get("query_image")
                query_sha = None
                if args.exp_admission == "on" and current_query_image[0]:
                    # Identity of the picture actually sent: the sha256 of its bytes.
                    from memorium.experience import image_sha256
                    query_sha = image_sha256(current_query_image[0])
                current_task[0] = raw_task
                task = TaskSpec(raw_task["question"], tuple(raw_task.get("options", ())),
                                constraints=tuple(c for c in (raw_task.get("format_instruction"),) if c),
                                metadata={"sample_id": unit_id, "question_id": raw_task["id"]})
                result = run_task(unit, task, raw_task["id"], arm, text_index=text_index, visual_index=visual_index,
                                  reader=reader, reranker=reranker, policy=policy,
                                  upgrade=materializer.upgrade if materializer else None,
                                  query_image=current_query_image[0],
                                  staging=exp,
                                  step=step, query_image_sha256=query_sha)
                if args.abstain_edit and not result.answer:
                    from memorium.evidence_edit import read_sheet, edit_sheet
                    sheet = edit_sheet(unit, task, result, arm)
                    if sheet is not None:
                        # A pointer-admitted experience node keeps its generated picture out here too.
                        pointer_nodes = pointers_of(result)
                        photos = [v for n in result.selected for v in unit.memory.nodes[n].views
                                  if v.kind in ("V0", "V3") and not (n in pointer_nodes and v.kind == "V3")][:4]
                        ans, cited, call = read_sheet(unit, task, [sheet] + photos, reader)
                        result.answer, result.cited = ans, cited
                        result.raw = str(call.get("output", ""))
                        result.reader_calls.append(call)
                expinfo = None
                if exp is not None:
                    adopted = exp.credit(raw_task["question"], result.answer, result.selected, result.cited, step,
                                         pointers=pointers_of(result))
                    node = exp.record(raw_task["question"], result.answer, result, step, args.routes,
                                      query_image_sha256=query_sha)
                    promoted = exp.consolidate(step) if (step + 1) % exp.s.period == 0 else []
                    expinfo = {"node": node, "adopted": adopted, "promoted": promoted,
                               "staged_hits": [n for n in result.selected if n in exp.records]}
                    if exp.s.admission:
                        # Three separate logs: ranked candidates, the pre-reader admission, post-answer adoption
                        # ("adopted" above; pointer reads that cited the node's records are listed apart).
                        expinfo["admission"] = result.experience_admission
                        expinfo["pointer_cited"] = list(exp.last_pointer_cited)
                        expinfo["query_image_sha256"] = query_sha
                last = result.reader_calls[-1] if result.reader_calls else {}
                (warm_out if raw_task.get("warmup") else rows).write(json.dumps({
                    "task_id": result.task_id, "sample_id": unit_id, "question": raw_task["question"],
                    "answer": result.answer, "raw": result.raw,
                    "source_refs": sorted({ref for node in result.selected
                                           for ref in unit.memory.nodes[node].source_refs}),
                    "selected": result.selected, "views": result.views, "retrieved_count": len(result.retrieved),
                    "memory_text_tokens": result.text_tokens, "memory_image_tokens": result.image_tokens,
                    "reader_input_tokens": sum(c.get("input_tokens", 0) for c in result.reader_calls),
                    "reader_vision_tokens": sum(c.get("input_vision_tokens", 0) for c in result.reader_calls),
                    "iterations": result.iterations, "note": result.note, "note_updated": result.note_updated,
                    "note_text": (unit.memory.nodes[result.note].view("T0").content if result.note else None),
                    "upgrades": result.upgrades, "cited": result.cited, "experience": expinfo,
                    "budget_halvings": result.budget_halvings,
                    "status": "OK" if all(c.get("status") == "OK" for c in result.reader_calls) else "FAILED",
                }, ensure_ascii=False, sort_keys=True) + "\n")
                # Lustre reports multi-MB blocks, so Python would otherwise buffer hundreds of rows in memory.
                rows.flush()
                warm_out.flush()
                total += 0 if raw_task.get("warmup") else 1
            kinds = defaultdict(int)
            views = defaultdict(int)
            for node in unit.memory.nodes.values():
                kinds[node_kind(node)] += 1
                for view in node.views:
                    views[view.kind] += 1
            write_json(output / "units" / f"{unit_id}.json", {
                "unit": unit_id, "tasks": len(tasks), "seconds": time.perf_counter() - unit_started,
                "final_revision": unit.memory.revision, "snapshot_hash": unit.memory.snapshot_hash,
                "nodes_by_kind": dict(kinds), "views_by_kind": dict(views),
                "v2_generated": unit.ledger.v2_generated,
                "experience": exp.summary() if exp is not None else None,
                "experience_events": exp.events if exp is not None else None,
            })
    manifest.update(status="TECHNICAL_PASS", tasks=total, seconds=time.perf_counter() - started,
                    finished_at=datetime.now(timezone.utc).isoformat())
    write_json(output / "run_manifest.json", manifest)
    print(json.dumps({"status": manifest["status"], "tasks": total, "units": len(unit_ids)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
