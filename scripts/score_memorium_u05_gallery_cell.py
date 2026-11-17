#!/usr/bin/env python3
"""Score a MEMORIUM U05 Mem-Gallery run with the pinned official Mem-Gallery evaluator.

Metrics come from the benchmark's own `evaluation.py` (exact_match_score, f1_score, bleu_score with 1-gram
weights), loaded by path and verified by sha256; the paid LLM judge is not used. Every question of the
requested units must be present. Reports per category and overall, plus evidence recall against the
official clue rounds.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path

EVALUATOR_SHA256 = "8ed8fb242ba0a9bf"  # prefix; full hash recorded in each report


def load_evaluator(path: Path):
    spec = importlib.util.spec_from_file_location("memgallery_pinned_evaluation", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--evaluator", type=Path, required=True)
    parser.add_argument("--units", required=True, help="comma-separated scenarios; all their questions must be present")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("--output must not already exist")
    sha = hashlib.sha256(args.evaluator.read_bytes()).hexdigest()
    if not sha.startswith(EVALUATOR_SHA256):
        raise ValueError(f"evaluator hash {sha} does not match the pinned {EVALUATOR_SHA256}")
    module = load_evaluator(args.evaluator)

    units = set(args.units.split(","))
    gold = {}
    for line in args.gold.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["scenario"] in units:
            gold[row["id"]] = row
    rows = {}
    for line in (args.run_root / "rows.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["task_id"] in rows:
            raise ValueError(f"duplicate task {row['task_id']}")
        rows[row["task_id"]] = row
    if set(rows) != set(gold):
        raise ValueError(f"rows/gold differ: missing {len(set(gold) - set(rows))}, extra {len(set(rows) - set(gold))}")

    per_row, buckets = [], defaultdict(lambda: defaultdict(float))
    for task_id in sorted(gold):
        qa, row = gold[task_id]["official_qa"], rows[task_id]
        prediction, answer = str(row.get("answer") or ""), str(qa["answer"])
        scores = {"em": float(module.exact_match_score(prediction, answer)),
                  "f1": float(module.f1_score(prediction, answer)),
                  "bleu1": float(module.bleu_score(prediction, answer, weights=(1, 0, 0, 0)))}
        clues = {f"memgallery:{row['sample_id']}:{c}" for c in qa.get("clue") or []}
        recall = len(clues & set(row.get("source_refs") or [])) / len(clues) if clues else None
        per_row.append({"task_id": task_id, "category": qa["point"], "prediction": prediction, **scores,
                        "evidence_recall": recall})
        for key in (qa["point"], "ALL"):
            bucket = buckets[key]
            bucket["n"] += 1
            for name, value in scores.items():
                bucket[name] += value
            if recall is not None:
                bucket["recall"] += recall
                bucket["recall_n"] += 1

    def summary(bucket):
        n = bucket["n"]
        return {"n": int(n), "em": bucket["em"] / n, "f1": bucket["f1"] / n, "bleu1": bucket["bleu1"] / n,
                "evidence_recall": bucket["recall"] / bucket["recall_n"] if bucket["recall_n"] else None}

    report = {"schema": "memorium-u05-gallery-score-v1", "run_root": str(args.run_root.resolve()),
              "units": sorted(units), "metric_authority": {"path": str(args.evaluator), "sha256": sha,
                                                          "functions": ["exact_match_score", "f1_score", "bleu_score(1,0,0,0)"],
                                                          "llm_judge": "not used"},
              "overall": summary(buckets["ALL"]),
              "per_category": {k: summary(v) for k, v in sorted(buckets.items()) if k != "ALL"}}
    args.output.mkdir(parents=True)
    (args.output / "score_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (args.output / "per_row.jsonl").write_text("".join(json.dumps(r) + "\n" for r in per_row), encoding="utf-8")
    print(json.dumps(report["overall"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
