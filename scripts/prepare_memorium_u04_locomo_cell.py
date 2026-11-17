#!/usr/bin/env python3
"""Prepare the LoCoMo dialogue and question files from the official release."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import re
from pathlib import Path
import sys
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Reused rather than reimplemented: the category-5 option order must be the official one.
from memorium.runtime.locomo import choose_options

FORBIDDEN = {"answer", "adversarial_answer", "category", "evidence", "gold", "label", "rationale", "score"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected object")
            rows.append(value)
    return rows


def key_names(value: Any) -> set[str]:
    if isinstance(value, dict):
        return {str(key) for key in value} | set().union(*(key_names(item) for item in value.values()))
    if isinstance(value, list):
        return set().union(*(key_names(item) for item in value)) if value else set()
    return set()


def assert_public(value: Any, where: str) -> None:
    leaked = sorted({key.lower() for key in key_names(value)} & FORBIDDEN)
    if leaked:
        raise ValueError(f"{where}: forbidden evaluation keys: {leaked}")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def source_rows(writer_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, dict[str, int]]]:
    records: list[dict[str, Any]] = []
    conversations: dict[str, dict[str, int]] = {}
    ordinal = 0
    for conversation in writer_rows:
        sample_id = conversation.get("sample_id")
        sessions = conversation.get("sessions")
        if not isinstance(sample_id, str) or not sample_id or not isinstance(sessions, list):
            raise ValueError("LoCoMo writer input has an invalid conversation")
        if sample_id in conversations:
            raise ValueError(f"duplicate conversation: {sample_id}")
        session_count = 0
        turn_count = 0
        visual_count = 0
        seen_turn_ids: set[str] = set()
        for session in sessions:
            session_index = session.get("index")
            date_time = session.get("date_time")
            turns = session.get("turns")
            if type(session_index) is not int or not isinstance(date_time, str) or not isinstance(turns, list):
                raise ValueError(f"{sample_id}: invalid session")
            session_count += 1
            for turn_index, turn in enumerate(turns):
                assert_public(turn, f"{sample_id}/{session_index}/{turn_index}")
                dia_id = turn.get("dia_id")
                speaker = turn.get("speaker")
                text = turn.get("text")
                caption = turn.get("blip_caption")
                if (not isinstance(dia_id, str) or not isinstance(speaker, str)
                        or not isinstance(text, str) or not text.strip()):
                    raise ValueError(f"{sample_id}: invalid turn")
                if dia_id in seen_turn_ids:
                    raise ValueError(f"{sample_id}: duplicate dia_id {dia_id}")
                seen_turn_ids.add(dia_id)
                if caption is not None and (not isinstance(caption, str) or not caption.strip()):
                    raise ValueError(f"{sample_id}/{dia_id}: invalid visual caption")
                record = {
                    "id": f"locomo:{sample_id}:{dia_id}",
                    "ordinal": ordinal,
                    "text": text,
                    "images": [],
                    "locator": {
                        "dataset": "locomo",
                        "sample_id": sample_id,
                        "session_index": session_index,
                        "turn_index": turn_index,
                        "dia_id": dia_id,
                        "speaker": speaker,
                        "date_time": date_time,
                        "visual_caption": caption,
                        "visual_representation": "caption_only" if caption else "none",
                    },
                }
                records.append(record)
                ordinal += 1
                turn_count += 1
                visual_count += caption is not None
        conversations[sample_id] = {
            "sessions": session_count,
            "turns": turn_count,
            "visual_caption_turns": visual_count,
        }
    return records, conversations


def public_tasks(question_rows: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    tasks = []
    seen: set[str] = set()
    for row in question_rows:
        question_id = row.get("id")
        sample_id = row.get("sample_id")
        question_index = row.get("question_index")
        official = row.get("official_qa")
        if (not isinstance(question_id, str) or not question_id or question_id in seen
                or not isinstance(sample_id, str) or type(question_index) is not int
                or not isinstance(official, dict) or not isinstance(official.get("question"), str)
                or not official["question"].strip()):
            raise ValueError("LoCoMo question row is invalid")
        seen.add(question_id)
        # Only the public question is projected; official_qa never enters this file. The one
        # exception is the category-5 option pair, which the official protocol *shows* to the
        # model -- scoring a free-form answer against a two-way choice would not be LoCoMo.
        options = choose_options(row, seed) if official.get("category") == 5 else []
        tasks.append({
            "id": question_id,
            "sample_id": sample_id,
            "question_index": question_index,
            "question": official["question"],
            "options": options,
        })
    return tasks


def source_only(sample):
    conversation = sample['conversation']
    numbers = sorted(int(k.split('_')[1]) for k, v in conversation.items() if re.fullmatch(r'session_\d+', k) and isinstance(v, list))
    sessions, ids = [], set()
    for number in numbers:
        turns = []
        for raw in conversation[f'session_{number}']:
            assert raw['dia_id'] not in ids, 'duplicate dialogue ID'
            ids.add(raw['dia_id'])
            assert all(isinstance(raw[k], str) for k in ('dia_id', 'speaker', 'text'))
            turn = {k: raw[k] for k in ('dia_id', 'speaker', 'text')}
            if 'blip_caption' in raw:
                turn['blip_caption'] = raw['blip_caption']
            turns.append(turn)
        sessions.append({'index': number, 'date_time': conversation[f'session_{number}_date_time'], 'turns': turns})
    return {'sample_id': sample['sample_id'], 'speaker_a': conversation['speaker_a'],
            'speaker_b': conversation['speaker_b'], 'sessions': sessions}, ids



def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--locomo10", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gold-output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=270727)
    args = parser.parse_args()

    if args.output.exists() or args.gold_output.exists():
        parser.error("use new output paths")
    samples = json.loads(args.locomo10.read_text(encoding="utf-8"))
    writer_rows, gold = [], []
    for sample in samples:
        writer, _ = source_only(sample)
        writer_rows.append(writer)
        unit = sample["sample_id"]
        for index, qa in enumerate(sample["qa"]):
            gold.append({"id": f"locomo:{unit}:{index:04d}", "sample_id": unit,
                         "question_index": index, "official_qa": qa})
    sources, counts = source_rows(writer_rows)
    tasks = public_tasks(gold, args.seed)
    if (len(counts), len(sources), len(tasks)) != (10, 5882, 1986):
        parser.error("expected the official 10-conversation LoCoMo release")
    if {task["sample_id"] for task in tasks} != set(counts):
        parser.error("question and source units differ")

    args.output.mkdir(parents=True)
    write_jsonl(args.output / "source_records.jsonl", sources)
    write_jsonl(args.output / "public_tasks.jsonl", tasks)
    args.gold_output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.gold_output, gold)
    manifest = {"dataset": "locomo", "conversations": len(counts), "turns": len(sources),
                "questions": len(tasks), "data_sha256": sha256(args.locomo10), "seed": args.seed}
    (args.output / "cell_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
