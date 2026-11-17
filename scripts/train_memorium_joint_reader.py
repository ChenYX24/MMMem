#!/usr/bin/env python3
"""Train one Reader LoRA on self-generated QA from any subset of the five domains (Single / Joint / LODO).

Training inputs are built by the real reading flow on each GEN unit: text retrieval -> budgeted selection (native
photos and keyframes travel with their nodes) -> the exact prompt the evaluation runner sends. Items from related
task sequences are replayed in stream order through the staging queue (memorium.stream, arm M4) with a scripted
reader that answers with the item's self-generated answer, so later items see what the queue would have offered.
Only earlier items of the same unit can shape a later item's input. Official QA and DEV/test units are never read.

--replay-summary (reader-alignment C1, off by default): the experience nodes replayed to later items carry the
"Summary:" line that evaluation-time nodes carry. Its text is the summary target the editing recipe already builds
for that item (summary_target below: answer text plus cited records, or "The records do not mention this."), keyed
by the item's own step, so only LATER items of the same unit can see it. It draws no random numbers and leaves
example ids, order, images and targets unchanged; only earlier-experience text in later prompts changes.
Repair 2026-10-31: the store itself always runs with summary off (the OFF replay is the SSOT for retrieval, the
node's evidence cut, the reader budget, selection and promotion); the line is inserted into the experience text
only when the example prompt is assembled (with_replayed_summary). The first build set summary on in the store,
which also changed the staged-route embedding and the budgets, so 986/1229 changed prompts and 247 image lists
differed beyond the Summary insert (audit G1/G2 failed).
--dump-prompts writes prompts.jsonl (index, unit, step, prompt) for audits. Items with a "query_image" (self-made
visual QA, C2) get that picture first, before the memory pictures, as the evaluation runner orders them.

Budget: --examples total, split equally over the domains given (each domain contributes the same number of QA
items; a domain with fewer items is resampled and the repeat count is reported). Abstention recipe (free-style
domains only) follows r2: answerable option share 0.5, abstain share 0.25.

Abstention calibration (amend8, off by default): in the forms the domain's own GEN questions use
(memorium/training/abstain_calibration.py) -- a stated abstention reply (--stated-abstain-share of the unanswerable
free-style items lose their options and get the domain's instruction, target its reply; as many answerable items
without options get the same instruction) and an abstention option (--abstain-option-share of the choice items get
it, half in place of the correct option, half in place of a wrong one). Its draws use a separate random stream, so
item sampling and example order are those of the uncalibrated recipe.

--reader-backend minicpmv (MiniCPM-V 4.5): inputs come from memorium.readers.MiniCPMInputs, the code the
evaluation reader uses; LoRA on q/k/v/o of every attention layer of its language model (llm.*, not the vision
tower); the forward takes MiniCPM-V's input dict.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from memorium import TaskSpec
from memorium.u05 import OPTIONS_INSTRUCTION, READER_INSTRUCTION, ArmConfig, build_indices, run_task
from memorium.unit import UnitConfig, build_unit
from memorium.training.abstain_calibration import abstention_option, stated_abstention, with_abstention_option

NO_INFO = "No information available"
STYLE = {"locomo": "free", "gallery": "free", "loogle": "choice", "lvbench": "choice", "ovo": "choice"}
BUDGET = {"locomo": 3000, "gallery": 4000, "loogle": 4000, "lvbench": 3000, "ovo": 3000}
CHOICE_FORMAT = ("This is a multiple-choice question: answer with only the letter (A, B, C or D) of the correct "
                 "option. Never answer insufficient; if the memory is not enough, pick the most likely option.")
LOOGLE_SUFFIX = ("\nDo not explain. Your entire response must be one line beginning exactly "
                 "`The correct answer is ` followed only by the final answer requested above.")


EXP_HEADER = "Experience from an earlier task (task "
EXP_EVIDENCE = "Evidence (verbatim records):"


def replay_summary_line(said: str | None) -> str:
    """The Summary text ExperienceStore.record() would keep for a summarize() returning `said` (first line, 400)."""
    lines = (said or "").strip().splitlines()
    return lines[0][:400] if lines else ""


def with_replayed_summary(text: str, summaries: dict[int, str]) -> str:
    """C1 repair (2026-10-31): insert "Summary: <s>" into an experience-node text built with summary off, exactly
    where record() puts it (just before the "Evidence (verbatim records):" line). Retrieval, the node's 384-token
    evidence cut, the reader budget, selection and the lifecycle are all those of the OFF replay; only this line is
    added. Text that is not an experience node of a step in `summaries` is returned unchanged."""
    first, _, _ = text.partition("\n")
    if not (first.startswith(EXP_HEADER) and first.endswith(")")):
        return text
    step = first[len(EXP_HEADER):-1]
    said = summaries.get(int(step)) if step.isdigit() else None
    lines = text.split("\n")
    if not said or EXP_EVIDENCE not in lines:
        return text
    k = lines.index(EXP_EVIDENCE)
    return "\n".join(lines[:k] + [f"Summary: {said}"] + lines[k:])


def summary_target(unit, item: dict, answer_text) -> tuple[list[str], str | None]:
    """The editing recipe's summary target for a verified item: (record lines it cites, target or None)."""
    ids = item["evidence"]
    lines = [unit.lines[n][i] for n in unit.segments for i in range(len(unit.lines[n]))
             if any(unit.lines[n][i].startswith(f"[{e}]") for e in ids)][:8]
    if lines:
        return lines, f"{answer_text(item['question'], item['answer'])} [{', '.join(ids)}]."
    if item["answer"].lower().startswith("no information"):
        return lines, "The records do not mention this."
    return lines, None


def sample_visual(ind: list[dict], need: int, wanted: int, rng) -> tuple[list[dict], int]:
    """Balance pictured QA across GEN units, with explicit resampling when data are scarce."""
    groups = defaultdict(list)
    for item in ind:
        if item.get("query_image"):
            groups[item["unit"]].append(item)
    if not groups:
        raise ValueError("visual sampling requested without pictured GEN QA")
    wanted = min(need, wanted)
    units = sorted(groups)
    visual = [rng.choice(groups[units[i % len(units)]]) for i in range(wanted)]
    ordinary = [item for item in ind if not item.get("query_image")]
    chosen = visual + ordinary[:need-wanted]
    while len(chosen) < need:
        chosen.append(rng.choice(ordinary or ind))
    return chosen, len(chosen) - len({id(item) for item in chosen})


def reader_evidence_ids(evidence: list[str], sources: dict) -> list[str]:
    """Use the same displayed citation IDs as unit.source_line, while retaining canonical source_refs."""
    result = []
    for ref in evidence:
        source = sources.get(ref)
        result.append((source.locator.get("dia_id") or source.id) if source is not None else ref)
    return result


def visible_evidence(item: dict, text: str) -> bool:
    """Keep positive supervision only when every cited source line is actually visible."""
    visible = {line[1:].split("]", 1)[0] for line in text.splitlines() if line.startswith("[") and "]" in line}
    return set(item.get("evidence", ())) <= visible


def main() -> int:
    model_root = Path(os.environ.get("MMMEM_WORKSPACE", Path.home() / "mmmem-workspace")) / "models"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", nargs="+", required=True, help="domain=path/to/rsi.jsonl (repeatable per domain)")
    parser.add_argument("--dev-split", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--examples", type=int, default=1500, help="total QA items (before abstention extras)")
    parser.add_argument("--sequence-share", type=float, default=0.5, help="share of items from related sequences")
    parser.add_argument("--shuffle-sequences", action="store_true", help="control: same items, sequence order lost")
    parser.add_argument("--model", default=str(model_root / 'Qwen3.5-9B'))
    parser.add_argument("--reader-backend", choices=("qwen35", "minicpmv"), default="qwen35")
    parser.add_argument("--text-encoder", default=str(model_root / 'Qwen3-Embedding-0.6B'))
    parser.add_argument("--visual-query-share", type=float, default=0.0,
                        help="QA quota reserved for pictured GEN items, balanced over their units")
    parser.add_argument("--grounded-inputs", action="store_true",
                        help="use the evaluation reranker, picture retrieval for visual QA, and filter absent evidence")
    parser.add_argument("--visual-encoder", default=str(model_root / 'Qwen3-VL-Embedding-2B'))
    parser.add_argument("--reranker", default=str(model_root / 'Qwen3-Reranker-0.6B'))
    parser.add_argument("--init-adapter", default=None, help="continue from an adapter (not used for LODO/Single)")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--accum", type=int, default=8)
    parser.add_argument("--seed", type=int, default=270727)
    parser.add_argument("--answerable-option-share", type=float, default=0.5)
    parser.add_argument("--abstain-share", type=float, default=0.25)
    parser.add_argument("--summary-share", type=float, default=0.25,
                        help="extra editing examples: write the experience-node summary for a verified item")
    parser.add_argument("--stated-abstain-share", type=float, default=0.0,
                        help="calibration: share of unanswerable free-style items asked with the domain's stated "
                             "abstention instruction instead of options")
    parser.add_argument("--abstain-option-share", type=float, default=0.0,
                        help="calibration: share of choice items given the domain's abstention option")
    parser.add_argument("--no-experience-replay", action="store_true",
                        help="build inputs without the experience store (answer-only control)")
    parser.add_argument("--replay-summary", action="store_true",
                        help="C1: replayed experience nodes carry the Summary line (summary_target of their item)")
    parser.add_argument("--dump-prompts", action="store_true", help="also write prompts.jsonl for audits")
    parser.add_argument("--examples-only", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.visual_query_share <= 1:
        parser.error("--visual-query-share must be between 0 and 1")
    if args.output.exists():
        parser.error("--output must not already exist")
    split = json.loads(args.dev_split.read_text(encoding="utf-8"))["domains"]
    for spec in split.values():
        spec["cell"] = os.path.expandvars(spec["cell"])
    rng = random.Random(args.seed)
    cal_rng = random.Random(f"{args.seed}:abstain-calibration")
    calibrate = args.stated_abstain_share > 0 or args.abstain_option_share > 0

    files = defaultdict(list)
    for spec in args.data:
        domain, path = spec.split("=", 1)
        if domain not in STYLE:
            parser.error(f"unknown domain {domain}")
        files[domain].append(Path(path))
    domains = sorted(files)
    quota = args.examples // len(domains)

    import torch
    minicpm = args.reader_backend == "minicpmv"
    if minicpm:
        from memorium.readers import MiniCPMInputs
        prep = MiniCPMInputs(args.model)

        def count_image(path: str) -> int:  # the evaluation runner's calls.count("", (path,))
            return prep.count("", (path,))
    else:
        from transformers import AutoProcessor
        from memorium.readers import configure_qwen_image_processor
        processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
        configure_qwen_image_processor(processor)

        def count_image(path: str) -> int:
            inputs = processor.apply_chat_template([{"role": "user", "content": [{"type": "image", "image": path},
                                                    {"type": "text", "text": ""}]}], add_generation_prompt=True,
                                                   enable_thinking=False, tokenize=True, return_dict=True, return_tensors="pt")
            return int(inputs["input_ids"].shape[-1])

    from memorium.adapters.route_index import QwenTextEncoder, RouteIndex
    from memorium.native import attach_native_views
    from memorium.experience import SUMMARY_PROMPT, ExperienceStore, answer_text, Settings as ExpSettings
    encoder = QwenTextEncoder(args.text_encoder)
    visual_encoder = visual_index = reranker = None
    if args.grounded_inputs:
        from memorium.training.reranker import load_reranker
        reranker = load_reranker(args.reranker)
    unsupported = Counter()

    args.output.mkdir(parents=True)
    examples, provenance, calibration = [], {}, {}
    for domain in domains:
        spec = split[domain]
        gen = set(spec["gen_units"])
        items = [json.loads(l) for p in files[domain] for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
        if any(i.get("warmup") for i in items):
            raise SystemExit(f"{domain}: warm-up rows (written from dev/test unit sources) must never be trained on")
        bad = {i["unit"] for i in items} - gen
        if bad:
            parser.error(f"{domain}: items from non-GEN units {sorted(bad)[:3]}")
        seq = [i for i in items if i.get("sequence_id")]
        ind = [i for i in items if not i.get("sequence_id")]
        n_seq = min(len(seq), round(quota * args.sequence_share)) if seq else 0
        # keep whole sequences together
        by_seq = defaultdict(list)
        for i in seq:
            by_seq[i["sequence_id"]].append(i)
        seq_ids = sorted(by_seq)
        rng.shuffle(seq_ids)
        chosen_seq = []
        for sid in seq_ids:
            if len(chosen_seq) >= n_seq:
                break
            chosen_seq += sorted(by_seq[sid], key=lambda i: i.get("position", 0))
        need = quota - len(chosen_seq)
        rng.shuffle(ind)
        repeats = 0
        chosen_ind = ind[:need]
        if args.visual_query_share and any(i.get("query_image") for i in ind):
            chosen_ind, repeats = sample_visual(ind, need, round(quota * args.visual_query_share), rng)
        while len(chosen_ind) < need and ind:
            chosen_ind.append(rng.choice(ind))
            repeats += 1
        provenance[domain] = {"items_available": len(items), "independent": len(chosen_ind),
                              "sequence_items": len(chosen_seq), "resampled": repeats,
                              "files": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files[domain]}}

        style = STYLE[domain]
        units_needed = sorted({i["unit"] for i in chosen_ind + chosen_seq})
        from scripts.run_memorium_joint_rsi import load_sources
        sources = load_sources(Path(spec["cell"]), set(units_needed))
        manifest = json.loads((Path(spec["cell"]) / "cell_manifest.json").read_text()) if (Path(spec["cell"]) / "cell_manifest.json").exists() else {}
        system_prompt = manifest.get("system_prompt", "")
        arm = ArmConfig(routes=("T",), write_back=False, token_budget=BUDGET[domain])
        stated = option = None
        if calibrate:  # question text of GEN units only, as few-shot reads it; never answers
            gen_tasks = [t for t in map(json.loads, (Path(spec["cell"]) / "public_tasks.jsonl").read_text(
                encoding="utf-8").splitlines()) if t["sample_id"] in gen]
            stated = stated_abstention(gen_tasks) if style == "free" and args.stated_abstain_share > 0 else None
            option = abstention_option(gen_tasks) if style == "choice" and args.abstain_option_share > 0 else None
        calibration[domain] = {"stated": list(stated) if stated else None, "option": option, "rewritten": Counter()}
        owed = 0  # answerable items still to receive the stated instruction

        for unit_id in units_needed:
            unit = build_unit(unit_id, sources[unit_id], UnitConfig(structure=False))
            if any(unit.memory.sources[r].images for r in unit.memory.sources):
                attach_native_views(unit, directory=args.output / "native", count_image=count_image, max_side=640)
            index = RouteIndex(encoder)
            pictured = args.grounded_inputs and any(i.get("query_image") and i["unit"] == unit_id
                                                     for i in chosen_ind + chosen_seq)
            visual_index = None
            if pictured:
                if visual_encoder is None:
                    from memorium.adapters.route_index import QwenVisualEncoder
                    visual_encoder = QwenVisualEncoder(args.visual_encoder)
                visual_index = RouteIndex(visual_encoder)
            build_indices(unit, index, visual_index)
            stream_items = [i for i in chosen_ind if i["unit"] == unit_id]
            seq_items = [i for i in chosen_seq if i["unit"] == unit_id]
            if args.shuffle_sequences:
                stream_items += seq_items
                rng.shuffle(stream_items)
            else:
                # sequences keep their internal order; unrelated items are interleaved between their steps
                rng.shuffle(stream_items)
                merged, pending = [], list(seq_items)
                while stream_items or pending:
                    if pending and (not stream_items or rng.random() < 0.5):
                        merged.append(pending.pop(0))
                    else:
                        merged.append(stream_items.pop(0))
                stream_items = merged
            # The experience store replays the real lifecycle: every earlier item leaves an experience node, staged
            # nodes are searched first, adopted ones are promoted; later items see what the system would show.
            ws = None if args.no_experience_replay else ExperienceStore(
                unit, RouteIndex(encoder), ExpSettings(image="off", summary=False), text_index=index)
            replayed = {}  # C1: step -> Summary text of that step's node, inserted only into later prompts' text
            for step, item in enumerate(stream_items):
                item = {**item, "evidence": reader_evidence_ids(item["evidence"], unit.memory.sources)}
                options = tuple(item.get("options") or ())
                if (style == "free" and not options and item["type"] not in ("no_information", "near_miss")
                        and rng.random() < args.answerable_option_share):
                    pair = [item["answer"], NO_INFO]
                    rng.shuffle(pair)
                    options = tuple(pair)
                constraints = (CHOICE_FORMAT,) if style == "choice" and domain != "loogle" else ()
                question, answer, evidence, rewrite = item["question"], item["answer"], item["evidence"], None
                if stated and item["type"] in ("no_information", "near_miss"):
                    if cal_rng.random() < args.stated_abstain_share:
                        options, constraints, answer, rewrite = (), (stated[0],), stated[1], "stated_abstain"
                        owed += 1
                elif stated and not options and owed > 0:
                    constraints, rewrite, owed = (stated[0],), "stated_answerable", owed - 1
                if option and cal_rng.random() < args.abstain_option_share:
                    as_answer = cal_rng.random() < 0.5
                    text = with_abstention_option(question, answer, option, as_answer, cal_rng)
                    if text:
                        question, rewrite = text, "option_answer" if as_answer else "option_distractor"
                        evidence = [] if as_answer else evidence
                if rewrite:
                    calibration[domain]["rewritten"][rewrite] += 1
                task = TaskSpec(question, options, constraints=constraints)
                captured = {}

                def reader(prompt, views, _item=item, _c=captured):
                    _c["prompt"], _c["views"] = prompt, list(views)
                    return {"output": f"Answer: {_item['answer']}\nEvidence: {', '.join(_item['evidence'])}",
                            "status": "OK"}

                query_image = item.get("query_image") if args.grounded_inputs else None
                input_arm = replace(arm, routes=("T", "V"), preserve_query_image_rank=True) if query_image else arm
                result = run_task(unit, task, f"{unit_id}:{step}", input_arm, text_index=index, visual_index=visual_index,
                                  reader=reader, staging=ws, step=step, query_image=query_image, reranker=reranker)
                body_for_audit = "\n\n".join(v.content for v in captured["views"] if v.modality == "text")
                if args.grounded_inputs and not visible_evidence(item, body_for_audit):
                    unsupported[domain] += 1
                    continue  # Do not replay an oracle answer whose evidence the Reader never received.
                if ws is not None:
                    ws.credit(item["question"], item["answer"], result.selected, result.cited, step)
                    node_id = ws.record(item["question"], item["answer"], result, step, "T")
                    if args.replay_summary and node_id is not None:  # deterministic, no random draw, store untouched
                        said = replay_summary_line(summary_target(unit, item, answer_text)[1])
                        if said and ws._lines(unit.memory.nodes[node_id].source_refs):  # record()'s own condition
                            replayed[step] = said
                    if (step + 1) % ws.s.period == 0:
                        ws.consolidate(step)
                views = captured["views"]
                texts = [with_replayed_summary(v.content, replayed) if v.generator == "experience" else v.content
                         for v in views if v.modality == "text"]
                images = [v.content for v in views if v.modality == "image"]
                if item.get("query_image"):  # image-first: the question's picture, then memory pictures (runner order)
                    images = [item["query_image"]] + images
                body = "\n\n".join(texts)
                if domain == "loogle":
                    prompt = (f"Please read the following material and answer the question.\n\n{body}\n\n"
                              f"Question: {question}{LOOGLE_SUFFIX}")
                    target = f"The correct answer is {answer}"
                else:
                    instruction = captured["prompt"]
                    if item.get("query_image"):
                        instruction += "\n(The first image is the picture shown with the question.)"
                    prompt = f"{instruction}\n\nMemory:\n{body}"
                    if system_prompt:
                        prompt = f"{system_prompt}\n\n{prompt}"
                    target = f"Answer: {answer}\nEvidence: {', '.join(evidence)}"
                examples.append({"domain": domain, "unit": unit_id, "type": item["type"],
                                 "recipe": "sequence" if item.get("sequence_id") else "independent",
                                 "prompt": prompt, "images": images, "target": target, "options": bool(options),
                                 **({"calibration": rewrite} if rewrite else {}), "_step": step})
                if rng.random() < args.summary_share:
                    # Editing skill: the experience-node summary for this verified item, citing its records.
                    lines, said = summary_target(unit, item, answer_text)
                    if said is not None:
                        shown = lines or [l for n in result.selected[:2] for l in unit.lines.get(n, [])][:6]
                        target = said
                        examples.append({"domain": domain, "unit": unit_id, "type": "summary", "recipe": "edit",
                                         "prompt": SUMMARY_PROMPT.format(question=item["question"],
                                                                         answer=item["answer"],
                                                                         records="\n".join(shown)),
                                         "images": [], "target": target, "options": False, "_step": step})
                if style == "free" and item["type"] not in ("no_information", "near_miss") and rng.random() < args.abstain_share:
                    q = TaskSpec(item["question"], ())
                    examples.append({"domain": domain, "unit": unit_id, "type": "abstain_empty", "recipe": "abstain",
                                     "prompt": f"{READER_INSTRUCTION}\n\nQuestion: {q.public_query}\n\nMemory:\n",
                                     "images": [], "options": False,
                                     "target": f"Answer: insufficient\nNeed: {item['question']}", "_step": step})
    rng.shuffle(examples)
    with (args.output / "examples.jsonl").open("w", encoding="utf-8") as stream:
        for e in examples:
            stream.write(json.dumps({k: v for k, v in e.items() if k not in ("prompt", "_step")}) + "\n")
    if args.dump_prompts:
        with (args.output / "prompts.jsonl").open("w", encoding="utf-8") as stream:
            for n, e in enumerate(examples):
                stream.write(json.dumps({"i": n, "unit": e["unit"], "step": e["_step"], "prompt": e["prompt"]}) + "\n")
    summary = {"examples": len(examples), "by_domain": dict(Counter(e["domain"] for e in examples)),
               "by_recipe": dict(Counter(e["recipe"] for e in examples)),
               "with_images": sum(bool(e["images"]) for e in examples), "provenance": provenance,
               "replay_summary": args.replay_summary,
               "query_image_examples": sum(1 for e in examples if e["type"] == "visual_query")}
    if calibrate:
        summary["abstain_calibration"] = {
            "stated_abstain_share": args.stated_abstain_share, "abstain_option_share": args.abstain_option_share,
            "by_domain": {d: dict(c, rewritten=dict(c["rewritten"])) for d, c in calibration.items()}}
    if args.grounded_inputs:
        summary.update(grounded_inputs=True, visual_query_share=args.visual_query_share, absent_evidence_filtered=dict(unsupported),
                       visual_query_retrieval="TV + evaluation reranker + image-rank fusion")
    print(json.dumps(summary))
    if args.examples_only:
        (args.output / "report.json").write_text(json.dumps(summary, indent=2) + "\n")
        return 0
    del encoder, index, visual_encoder, visual_index, reranker, ws
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForMultimodalLM, get_cosine_schedule_with_warmup
    if minicpm:
        from memorium.readers import load_minicpmv
        model = load_minicpmv(args.model)
        language = "llm.model.layers."  # the SigLIP vision tower has q/k/v projections of its own
    else:
        model = AutoModelForMultimodalLM.from_pretrained(args.model, local_files_only=True, dtype=torch.bfloat16,
                                                         device_map={"": 0})
        language = "language_model"
    if args.init_adapter:
        model = PeftModel.from_pretrained(model, args.init_adapter, is_trainable=True)
    else:
        targets = [name for name, module in model.named_modules() if isinstance(module, torch.nn.Linear)
                   and language in name and name.rsplit(".", 1)[-1] in ("q_proj", "k_proj", "v_proj", "o_proj")]
        model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05,
                                                 target_modules=targets, task_type="CAUSAL_LM"))
    backbone = model.get_base_model().llm if minicpm else model
    backbone.config.use_cache = False
    backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    backbone.enable_input_require_grads()
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    updates = max(1, len(examples) // args.accum)
    scheduler = get_cosine_schedule_with_warmup(optimizer, max(1, updates // 10), updates)
    log = (args.output / "train_log.jsonl").open("w", encoding="utf-8")
    started, tokens = time.time(), Counter()
    trainable = sum(p.numel() for p in params)
    for step, example in enumerate(examples, 1):
        if minicpm:
            inputs = prep.inputs(example["prompt"], example["images"])
        else:
            content = [{"type": "image", "image": p} for p in example["images"]] + [{"type": "text", "text": example["prompt"]}]
            inputs = processor.apply_chat_template([{"role": "user", "content": content}], add_generation_prompt=True,
                                                   enable_thinking=False, tokenize=True, return_dict=True,
                                                   return_tensors="pt")
        length = inputs["input_ids"].shape[-1]
        if length > 12000:
            log.write(json.dumps({"step": step, "skipped": "too_long", "prompt_tokens": length}) + "\n")
            continue
        if minicpm:
            # Same tokens the reader sees, its prompt attention mask (the processor masks token id 0 at inference)
            # and the answer appended after the chat's empty think block; image_bound stays valid.
            target = prep.tokenizer(example["target"] + "<|im_end|>", add_special_tokens=False,
                                    return_tensors="pt")["input_ids"]
            inputs = inputs.to(0)
            inputs.pop("image_sizes", None)
            ids = torch.cat([inputs["input_ids"], target.to(0)], dim=1)
            mask = torch.cat([inputs["attention_mask"], torch.ones_like(target, dtype=inputs["attention_mask"].dtype).to(0)], dim=1)
            labels = ids.clone()
            labels[:, :length] = -100
            data = {"input_ids": ids, "pixel_values": inputs["pixel_values"], "tgt_sizes": inputs["tgt_sizes"],
                    "image_bound": inputs["image_bound"], "position_ids": torch.arange(ids.shape[1], device=ids.device)[None]}
            loss = model.get_base_model()(data=data, attention_mask=mask, labels=labels, use_cache=False).loss
        else:
            target = processor.tokenizer(example["target"] + processor.tokenizer.eos_token, add_special_tokens=False,
                                         return_tensors="pt")["input_ids"]
            inputs["input_ids"] = torch.cat([inputs["input_ids"], target], dim=1)
            inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])
            for key in ("mm_token_type_ids", "token_type_ids"):
                if key in inputs:
                    inputs[key] = torch.cat([inputs[key], torch.zeros_like(target)], dim=1)
            labels = inputs["input_ids"].clone()
            labels[:, :length] = -100
            loss = model(**{k: v.to(0) for k, v in inputs.items()}, labels=labels.to(0), use_cache=False).loss
        if not torch.isfinite(loss):
            raise ValueError("nonfinite training loss")
        (loss / args.accum).backward()
        tokens[example["domain"]] += length
        if step % args.accum == 0 or step == len(examples):
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
        log.write(json.dumps({"step": step, "loss": float(loss.detach()), "domain": example["domain"],
                              "prompt_tokens": length, "seconds": time.time() - started}) + "\n")
        log.flush()
    model.save_pretrained(args.output / "adapter")
    # Nodes map the shared disk to different uids; a 0600 adapter is unreadable from the evaluating node.
    for path in (args.output / "adapter").rglob("*"):
        path.chmod(0o644 if path.is_file() else 0o755)
    report = dict(summary, schema="memorium-joint-reader-lora-v1", domains=domains, quota_per_domain=quota,
                  sequence_share=args.sequence_share, summary_share=args.summary_share,
                  experience_replay=not args.no_experience_replay, shuffle_sequences=args.shuffle_sequences,
                  input_tokens_by_domain=dict(tokens), lora_rank=args.rank, lr=args.lr, accum=args.accum,
                  updates=updates, seed=args.seed, init_adapter=args.init_adapter, seconds=time.time() - started,
                  reader_backend=args.reader_backend, model=args.model, trainable_parameters=trainable,
                  adapter=str(args.output / "adapter"), dev_or_test_units_read=False, official_qa_read=False)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("examples", "by_domain", "seconds")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
