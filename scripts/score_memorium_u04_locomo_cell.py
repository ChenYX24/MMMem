#!/usr/bin/env python3
"""Score a completed MEMORIUM U04 LoCoMo cell with the pinned official evaluator."""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import redirect_stdout
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from memorium.runtime.locomo import official_scorer, parse_choice
from memorium.runtime.metrics import exact_match, token_f1

SCHEMA = "memorium-u04-locomo-score-v1"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_rows(run_root: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    shards = sorted(run_root.glob("shard_*/rows.jsonl")) or sorted(run_root.glob("rows.jsonl"))
    if not shards:
        raise ValueError(f"no shard rows under {run_root}")
    for path in shards:
        for row in read_jsonl(path):
            task_id = row.get("task_id")
            if task_id in rows:
                raise ValueError(f"duplicate task_id across shards: {task_id}")
            rows[task_id] = row
    return rows


def evidence_recall(gold_evidence: list[str], sample_id: str, source_refs: list[str]) -> float | None:
    if not gold_evidence:
        return None
    retrieved = set(source_refs or ())
    hit = sum(1 for dia_id in gold_evidence if f"locomo:{sample_id}:{dia_id}" in retrieved)
    return hit / len(gold_evidence)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--cell", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--evaluator", type=Path, required=True, help="official task_eval/evaluation.py")
    parser.add_argument("--units", default=None,
                        help="comma-separated sample_ids; every question of these units must be present")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("--output must not already exist")

    rows = load_rows(args.run_root.resolve())
    gold = {row["id"]: row for row in read_jsonl(args.gold.resolve())}
    cell_tasks = {task["id"]: task for task in read_jsonl(args.cell.resolve() / "public_tasks.jsonl")}
    units = sorted(args.units.split(",")) if args.units else None
    if units:
        gold = {key: row for key, row in gold.items() if row["sample_id"] in units}
        cell_tasks = {key: task for key, task in cell_tasks.items() if task["sample_id"] in units}
        if not gold:
            raise ValueError("no gold questions for the requested units")

    if set(rows) != set(gold):
        raise ValueError(f"rows/gold are not a bijection: rows={len(rows)} gold={len(gold)} "
                         f"missing={len(set(gold) - set(rows))} extra={len(set(rows) - set(gold))}")
    if set(rows) != set(cell_tasks):
        raise ValueError("rows and cell tasks disagree")

    module = official_scorer(args.evaluator)

    # Category 5 is a two-way choice; without the option pair the run answered a different task.
    category5 = [task_id for task_id in rows if gold[task_id]["official_qa"].get("category") == 5]
    protocol_match = bool(category5) and all(cell_tasks[task_id].get("options") for task_id in category5)

    ordered = sorted(rows)
    qas, records = [], []
    for task_id in ordered:
        row, official = rows[task_id], gold[task_id]["official_qa"]
        category = official.get("category")
        raw = row.get("answer") or ""
        options = cell_tasks[task_id].get("options") or []
        if category == 5 and options:
            prediction = parse_choice(raw, options)
        else:
            prediction = raw.strip()
        qa = {**official, "prediction": prediction or ""}
        qa.setdefault("answer", "")
        qas.append(qa)
        records.append({"task_id": task_id, "category": category, "raw": raw,
                        "prediction": prediction or "", "sample_id": row.get("sample_id"),
                        "evidence_recall": evidence_recall(official.get("evidence") or [],
                                                           row.get("sample_id") or "",
                                                           row.get("source_refs") or [])})

    # The pinned evaluator returns (all_ems, lengths, all_recall), not one entry per question.
    with redirect_stdout(io.StringIO()):
        all_ems, _, _ = module.eval_question_answering(qas)
    if len(all_ems) != len(qas):
        raise ValueError("official evaluator returned an unexpected number of scores")

    buckets: dict[Any, dict[str, float]] = defaultdict(
        lambda: {"n": 0, "official": 0.0, "em": 0.0, "f1": 0.0, "recall": 0.0, "recall_n": 0,
                 "retrieved_but_wrong": 0})
    for record, entry, qa in zip(records, all_ems, qas):
        score = float(entry)
        record["official_score"] = score
        bucket = buckets[record["category"]]
        bucket["n"] += 1
        bucket["official"] += score
        bucket["em"] += exact_match(record["prediction"], str(qa.get("answer", "")))
        bucket["f1"] += token_f1(record["prediction"], str(qa.get("answer", "")))
        if record["evidence_recall"] is not None:
            bucket["recall_n"] += 1
            bucket["recall"] += record["evidence_recall"]
            if record["evidence_recall"] == 1.0 and score == 0.0:
                bucket["retrieved_but_wrong"] += 1

    def summarise(categories: list[Any]) -> dict[str, Any]:
        picked = [buckets[category] for category in categories if category in buckets]
        n = sum(bucket["n"] for bucket in picked)
        recall_n = sum(bucket["recall_n"] for bucket in picked)
        if not n:
            return {"n": 0}
        return {
            "n": n,
            "official_score": sum(bucket["official"] for bucket in picked) / n,
            "evidence_recall": (sum(bucket["recall"] for bucket in picked) / recall_n) if recall_n else None,
            "retrieved_but_wrong": sum(bucket["retrieved_but_wrong"] for bucket in picked),
            "diagnostic_unofficial": {
                "exact_match": sum(bucket["em"] for bucket in picked) / n,
                "token_f1": sum(bucket["f1"] for bucket in picked) / n,
            },
        }

    headline_categories = [1, 2, 3, 4, 5] if protocol_match else [1, 2, 3, 4]
    report = {
        "schema": SCHEMA,
        "run_root": str(args.run_root.resolve()),
        "cell": str(args.cell.resolve()),
        "units": units or "all",
        "scored_at": datetime.now(timezone.utc).isoformat(),
        "metric_authority": {
            "path": str(args.evaluator.resolve()),
            "sha256": sha256_file(args.evaluator),
            "function": "eval_question_answering",
            "verified": True,
        },
        "protocol": {
            "category5_options_present": protocol_match,
            "headline_categories": headline_categories,
            "note": ("full official protocol" if protocol_match else
                     "category 5 ran free-form without its option pair, so it is excluded from the "
                     "headline denominator and reported as a diagnostic only"),
        },
        "headline": summarise(headline_categories),
        "per_category": {str(category): summarise([category]) for category in sorted(buckets, key=str)},
        "all_categories": summarise(sorted(buckets, key=str)),
    }

    args.output.mkdir(parents=True)
    (args.output / "score_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (args.output / "per_row.jsonl").open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps(report["headline"], ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
