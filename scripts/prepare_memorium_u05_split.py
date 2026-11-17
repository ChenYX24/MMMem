#!/usr/bin/env python3
"""Split memory units (never questions) into train/test, stratified by question count.

Units share facts across their questions, so a question-level split would leak test answers into
training. The split is deterministic given the seed and is committed before any training run.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def split_units(counts: dict[str, int], n_train: int, seed: int) -> tuple[list[str], list[str]]:
    """Sort units by question count, walk them in pairs/strata, and pick train units by a seeded hash.

    Stratifying on size keeps the train and test halves from differing systematically in unit length.
    """
    if not 0 < n_train < len(counts):
        raise ValueError("train size must leave at least one unit on each side")
    ordered = sorted(counts, key=lambda unit: (counts[unit], unit))
    strata = max(1, n_train)
    buckets = [ordered[i::strata] for i in range(strata)]
    train: list[str] = []
    for bucket in buckets:
        if len(train) >= n_train:
            break
        ranked = sorted(bucket, key=lambda unit: hashlib.sha256(f"{seed}:{unit}".encode()).hexdigest())
        train.append(ranked[0])
    rest = sorted((u for u in ordered if u not in train),
                  key=lambda unit: hashlib.sha256(f"{seed}:{unit}".encode()).hexdigest())
    while len(train) < n_train:
        train.append(rest.pop(0))
    test = [unit for unit in ordered if unit not in train]
    return sorted(train), sorted(test)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True, help="public_tasks.jsonl of the cell")
    parser.add_argument("--n-train", type=int, required=True)
    parser.add_argument("--seed", type=int, default=270727)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("--output must not already exist")
    counts: Counter[str] = Counter()
    for line in args.tasks.read_text(encoding="utf-8").splitlines():
        if line.strip():
            counts[json.loads(line)["sample_id"]] += 1
    train, test = split_units(dict(counts), args.n_train, args.seed)
    payload = {
        "schema": "memorium-u05-split-v1", "dataset": args.dataset, "unit": "memory_unit (sample_id)",
        "seed": args.seed, "train": train, "test": test,
        "questions": {"train": sum(counts[u] for u in train), "test": sum(counts[u] for u in test)},
        "per_unit_questions": dict(sorted(counts.items())),
        "tasks_sha256": hashlib.sha256(args.tasks.read_bytes()).hexdigest(),
        "rule": "train units supply source text for self-generated (RSI) data and their official QA as dev only; "
                "test units are never read during training",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: payload[k] for k in ("train", "test", "questions")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
