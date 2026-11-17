#!/usr/bin/env python3
"""Score a MEMORIUM U05 LooGLE-v2 run with the repository's pinned LooGLE-v2 judge.

The answer is extracted from the reader's raw reply with `extract_answer` (the official "The correct answer
is ..." contract) and judged with `judge_answer`/`judge_score` (Finance 5% tolerance, Version Control
Jaccard), copied verbatim into memorium/adapters/loogle.py. Every question of the requested documents
must be present. Reports overall, per domain, per task, and on multi-question documents.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from memorium.adapters.loogle import extract_answer, judge_answer, judge_score  # noqa: E402


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True, help="private_eval/gold.jsonl")
    parser.add_argument("--tasks", type=Path, required=True, help="cell public_tasks.jsonl (id -> document)")
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--part", choices=("train", "test"), default="test")
    parser.add_argument("--units", default=None, help="comma-separated documents; default: the whole split part")
    parser.add_argument("--partial", action="store_true",
                        help="smoke runs only: score the rows present; the report is marked partial")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("--output must not already exist")
    units = set(args.units.split(",")) if args.units else set(json.loads(args.split.read_text())[args.part])
    document = {t["id"]: t["sample_id"] for t in read_jsonl(args.tasks)}
    per_doc = Counter(document.values())
    gold = {g["id"]: g for g in read_jsonl(args.gold) if document.get(g["id"]) in units}
    rows = {}
    for row in read_jsonl(args.run_root / "rows.jsonl"):
        if row["task_id"] in rows:
            raise ValueError(f"duplicate task {row['task_id']}")
        rows[row["task_id"]] = row
    if args.partial:
        gold = {k: v for k, v in gold.items() if k in rows}
    if set(rows) != set(gold):
        raise ValueError(f"rows/gold differ: missing {len(set(gold) - set(rows))}, extra {len(set(rows) - set(gold))}")

    per_row, buckets = [], defaultdict(lambda: [0.0, 0, 0])
    for task_id in sorted(gold):
        g, raw = gold[task_id], str(rows[task_id].get("raw") or "")
        prediction = extract_answer(raw, g["task"], g["source"])
        score = judge_score(judge_answer(prediction, g["answer"], g["task"]))
        multi = per_doc[document[task_id]] > 1
        per_row.append({"task_id": task_id, "document": document[task_id], "domain": g["source"], "task": g["task"],
                        "prediction": prediction, "score": score, "parsed": prediction is not None,
                        "multi_question_doc": multi})
        for key in ("ALL", f"domain:{g['source']}", f"task:{g['task']}", *(("multi_question_docs",) if multi else ())):
            bucket = buckets[key]
            bucket[0] += score
            bucket[1] += 1
            bucket[2] += prediction is not None

    summary = {k: {"n": n, "score": s / n, "parse_rate": p / n} for k, (s, n, p) in sorted(buckets.items())}
    report = {"schema": "memorium-u05-loogle-score-v1", "run_root": str(args.run_root.resolve()),
              "units": len(units), "partial": args.partial, "metric_authority": "memorium/adapters/loogle.py (LooGLE v2 4365a5c semantics)",
              "overall": summary.pop("ALL"), "breakdown": summary}
    args.output.mkdir(parents=True)
    (args.output / "score_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (args.output / "per_row.jsonl").write_text("".join(json.dumps(r) + "\n" for r in per_row), encoding="utf-8")
    print(json.dumps(report["overall"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
