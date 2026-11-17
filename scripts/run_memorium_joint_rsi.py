#!/usr/bin/env python3
"""Generate synthetic QA from source records of one benchmark split.

Round 1 uses the frozen base Reader. Round 2 passes --adapter: the R1 adapter is really loaded (ModelCalls with
adapter_path, every call runs as stage "answerer" through it) and its file hashes are recorded in the report.
Source ids are the cell's own ids (any dataset); segment images travel with the generator call.
Refuses DEV and test units. Official QA is never read.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from memorium import SourceRecord
from memorium.training.rsi_multi import FREE_IMAGE_NOTE, VIDEO_NOTE, generate
from memorium.unit import UnitConfig, build_unit

STYLE = {"locomo": "free", "gallery": "free", "loogle": "choice"}


def load_sources(cell: Path, units: set[str]) -> dict[str, list[SourceRecord]]:
    by_unit = defaultdict(list)
    with (cell / "source_records.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            unit = row["locator"]["sample_id"]
            if unit in units:
                by_unit[unit].append(SourceRecord(row["id"], row["ordinal"], row["text"], row["locator"],
                                                  images=tuple(row.get("images") or ())))
    return by_unit


def main() -> int:
    model_root = Path(os.environ.get("MMMEM_WORKSPACE", Path.home() / "mmmem-workspace")) / "models"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", choices=sorted(STYLE), required=True)
    parser.add_argument("--dev-split", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--recipe", choices=("independent", "sequence"), default="independent")
    parser.add_argument("--segments", type=int, default=300, help="segments sampled across GEN units")
    parser.add_argument("--per-segment", type=int, default=4)
    parser.add_argument("--shard", default="0/1", help="i/n: this process handles every n-th sampled segment")
    parser.add_argument("--round", type=int, default=1)
    parser.add_argument("--model", default=str(model_root / 'Qwen3.5-9B'))
    parser.add_argument("--reader-backend", choices=("qwen35", "minicpmv"), default="qwen35",
                        help="the frozen backbone that writes and checks the questions (round 1 and warm-up)")
    parser.add_argument("--adapter", default=None, help="round >= 2: generate and check with this adapter loaded")
    parser.add_argument("--frozen-control", action="store_true",
                        help="round >= 2 non-recursive control: generate with the frozen base model, same budget/seed")
    parser.add_argument("--fewshot", type=int, default=0,
                        help="show this many official questions of GEN units (question text only, never answers)")
    parser.add_argument("--experience-dir", type=Path, default=None,
                        help="also generate from long-term experience nodes dumped by a previous round's run")
    parser.add_argument("--warmup", choices=("dev", "test"), default=None,
                        help="write warm-up questions for the DEV or TEST units of the split from their own source "
                             "records (no official QA of those units is read; few-shot uses GEN units only). "
                             "Rows are marked warmup=true and the trainer refuses them.")
    parser.add_argument("--segments-per-unit", type=int, default=None,
                        help="with --warmup: segments sampled in every unit instead of --segments in total")
    parser.add_argument("--max-images", type=int, default=3)
    parser.add_argument("--seed", type=int, default=270727)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("--output must not already exist")
    if args.round >= 2 and not (args.adapter or args.dry_run or args.frozen_control):
        parser.error("round >= 2 must generate with the previous round's adapter")
    if args.reader_backend != "qwen35" and args.adapter:
        parser.error("--adapter generation is implemented for qwen35 only")
    spec = json.loads(args.dev_split.read_text(encoding="utf-8"))["domains"][args.domain]
    spec["cell"] = os.path.expandvars(spec["cell"])
    gen = set(spec["gen_units"])
    if gen & set(spec["dev_units"]):
        parser.error("GEN and DEV units overlap")
    split = json.loads(Path(spec["split"]).read_text(encoding="utf-8"))
    if gen & set(split["test"]) or not gen <= set(split["train"]):
        parser.error("GEN units must be train units of the frozen split")
    target = gen
    if args.warmup:
        target = set(spec["dev_units"]) if args.warmup == "dev" else set(split["test"])
        if args.experience_dir:
            parser.error("--warmup writes from source records only; --experience-dir is for training rounds")
    style = STYLE[args.domain]
    note = VIDEO_NOTE if args.domain in ("lvbench", "ovo") else FREE_IMAGE_NOTE if args.domain == "gallery" else ""
    sources = load_sources(Path(spec["cell"]), target)

    rng = random.Random(f"{args.seed}:{args.domain}:{args.recipe}:{args.round}")
    candidates = []
    units = {}
    for unit_id in sorted(sources):
        unit = build_unit(unit_id, sources[unit_id], UnitConfig(structure=False))
        units[unit_id] = unit
        candidates += [(unit_id, seg) for seg in unit.segments]
    rng.shuffle(candidates)
    # Round-robin across units so one long document does not dominate.
    per_unit = defaultdict(list)
    for unit_id, seg in candidates:
        per_unit[unit_id].append(seg)
    picked, order = [], sorted(per_unit)
    quota = args.segments_per_unit * len(order) if args.warmup and args.segments_per_unit else args.segments
    taken = Counter()
    while len(picked) < quota and any(per_unit.values()):
        for unit_id in order:
            if per_unit[unit_id] and len(picked) < quota and (
                    not (args.warmup and args.segments_per_unit) or taken[unit_id] < args.segments_per_unit):
                picked.append((unit_id, per_unit[unit_id].pop()))
                taken[unit_id] += 1
            elif args.warmup and args.segments_per_unit:
                per_unit[unit_id] = []
    # Experience nodes of a previous round become extra generation segments (their quoted evidence lines).
    exp_segments = {}
    if args.experience_dir:
        for path in sorted(args.experience_dir.glob("*.jsonl")):
            for line in path.read_text(encoding="utf-8").splitlines():
                node = json.loads(line)
                if node["unit"] in units and node.get("state") == "long_term":
                    lines = [l for l in node["text"].splitlines() if l.startswith("[")]
                    if lines:
                        exp_segments[(node["unit"], node["id"])] = (lines, node.get("images") or [])
        picked += [(u, nid) for (u, nid) in sorted(exp_segments)][:max(0, args.segments // 2)]
    pool_questions = []
    if args.fewshot:
        for line in (Path(spec["cell"]) / "public_tasks.jsonl").read_text(encoding="utf-8").splitlines():
            task = json.loads(line)
            if task["sample_id"] in gen:  # never the questions of the unit being warmed up
                pool_questions.append(task["question"].strip())
    shard, shards = map(int, args.shard.split("/"))
    picked = picked[shard::shards]

    adapter_hashes = None
    if args.dry_run:
        call = lambda prompt, images, max_new: "[]"
    elif args.reader_backend == "minicpmv":
        from memorium.readers import make_calls
        calls = make_calls(args.model, "minicpmv", args.output / "calls", max_input_tokens=32768)
        call = lambda prompt, images, max_new: str(calls.call("writer", prompt, images, max_new_tokens=max_new).get("output", ""))
    else:
        from memorium.runtime.inference import ModelCalls
        calls = ModelCalls(args.model, "qwen35", args.output / "calls", max_input_tokens=32768,
                           adapter_path=args.adapter)
        adapter_hashes = getattr(calls, "adapter_hashes", None)
        stage = "answerer" if args.adapter else "writer"
        call = lambda prompt, images, max_new: str(calls.call(stage, prompt, images, max_new_tokens=max_new).get("output", ""))

    args.output.mkdir(parents=True, exist_ok=True)
    tally, kinds, rows = Counter(), Counter(), 0
    with (args.output / "rsi.jsonl").open("w", encoding="utf-8") as stream:
        for unit_id, segment in picked:
            unit = units[unit_id]
            if (unit_id, segment) in exp_segments:
                seg_lines, images = exp_segments[(unit_id, segment)]
                images = images[:args.max_images]
                refs = [r for r in unit.memory.sources if any(l.startswith(f"[{unit.memory.sources[r].locator.get('dia_id') or r}]") for l in seg_lines)]
            else:
                node = unit.memory.nodes[segment]
                seg_lines, refs = unit.lines[segment], node.source_refs
                images = [img for ref in refs for img in unit.memory.sources[ref].images][:args.max_images]
            shots = rng.sample(pool_questions, min(args.fewshot, len(pool_questions))) if pool_questions else []
            items, counts = generate(seg_lines, images, style, args.recipe, call, n=args.per_segment,
                                     rng=rng, note=note if images or args.domain in ("lvbench", "ovo") else "",
                                     unit=unit_id, segment=segment, fewshot=shots, round=args.round)
            tally.update(counts)
            lookup = {str(unit.memory.sources[r].locator.get("dia_id") or r): r for r in refs}
            for item in items:
                item.update(domain=args.domain, round=args.round, fewshot=len(shots), warmup=bool(args.warmup),
                            from_experience=(unit_id, segment) in exp_segments,
                            evidence_sources=[lookup[e] for e in item["evidence"] if e in lookup],
                            generator_images=images, adapter=args.adapter)
                kinds[item["type"]] += 1
                stream.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
                rows += 1
            stream.flush()
    report = {"schema": "memorium-joint-rsi-v1", "domain": args.domain, "recipe": args.recipe, "round": args.round,
              "style": style, "gen_units": sorted(gen), "segments": len(picked), "shard": args.shard,
              "dev_split_sha256": hashlib.sha256(args.dev_split.read_bytes()).hexdigest(),
              "items": rows, "by_type": dict(kinds), "filter_tally": dict(tally), "model": args.model,
              "reader_backend": args.reader_backend,
              "adapter": args.adapter, "adapter_hashes": adapter_hashes, "fewshot": args.fewshot,
              "experience_segments": len(exp_segments), "fewshot_pool": len(pool_questions), "frozen_control": args.frozen_control, "official_qa_read": False, "warmup": args.warmup,
              "target_units": sorted(target), "source_records_of_dev_or_test_read": bool(args.warmup),
              "dev_or_test_units_read": bool(args.warmup), "dry_run": args.dry_run}
    (args.output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("domain", "items", "by_type", "filter_tally")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
