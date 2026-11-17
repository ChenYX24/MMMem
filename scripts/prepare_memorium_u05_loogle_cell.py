#!/usr/bin/env python3
"""Prepare LooGLE v2 source records and questions from its Parquet release."""
from __future__ import annotations

import argparse
import ast
from collections import defaultdict
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from memorium.adapters.loogle import option_text


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pack(text: str, limit: int) -> list[str]:
    """Split on newlines, then on spaces for overlong lines; concatenating the pieces restores the text."""
    pieces, current = [], ""
    for line in text.splitlines(keepends=True):
        while len(line) > limit:
            cut = line.rfind(" ", 0, limit)
            cut = limit if cut <= 0 else cut + 1
            if current:
                pieces.append(current)
                current = ""
            pieces.append(line[:cut])
            line = line[cut:]
        if current and len(current) + len(line) > limit:
            pieces.append(current)
            current = ""
        current += line
    if current:
        pieces.append(current)
    return [piece for piece in pieces if piece.strip()]


def options_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return []
        try:
            parsed = ast.literal_eval(value)
            return [str(v) for v in parsed] if isinstance(parsed, (list, tuple)) else [value]
        except (ValueError, SyntaxError):
            return [value]
    return [str(v) for v in value]


def load_release(paths):
    """Collect public questions, deduplicated source documents, and scorer-only labels."""
    from pyarrow.parquet import ParquetFile

    by_doc = defaultdict(list)
    contexts, gold, ids = {}, [], set()
    fields = ("id", "context", "question", "instruction", "options", "source", "task", "type", "answer")
    for path in sorted(paths):
        for batch in ParquetFile(path).iter_batches(batch_size=8, columns=list(fields)):
            for row in batch.to_pylist():
                case_id = str(row["id"])
                if case_id in ids:
                    raise ValueError(f"duplicate official question ID: {case_id}")
                ids.add(case_id)
                context = row["context"]
                document = hashlib.sha256(context.encode("utf-8")).hexdigest()
                contexts.setdefault(document, context)
                options = option_text(row["options"])
                retrieval_query = row["question"] + ("\nOptions:\n" + options if options else "")
                by_doc[document].append({
                    "id": case_id, "sample_id": document, "question_index": 0,
                    "question": retrieval_query, "options": [], "loogle_question": row["question"],
                    "loogle_options": options_list(options), "instruction": row["instruction"],
                    "task": row["task"], "source": row["source"], "type": row["type"],
                })
                gold.append({"id": case_id, "answer": row["answer"],
                             "task": row["task"], "source": row["source"]})
    return contexts, by_doc, gold


def public_tasks(by_doc):
    for document in sorted(by_doc):
        for index, row in enumerate(sorted(by_doc[document], key=lambda task: task["id"])):
            yield {**row, "question_index": index}


def source_records(contexts):
    """Use chronological 8,000-character chunks and lossless paragraph packing."""
    ordinal = 0
    for document in sorted(contexts):
        context = contexts[document]
        for sequence, start in enumerate(range(0, len(context), 8000)):
            chunk = f"s{sequence:05d}"
            value = context[start:start + 8000]
            paragraphs = pack(value, 1500)
            if re.sub(r"\s", "", value) != re.sub(r"\s", "", "".join(paragraphs)):
                raise ValueError(f"paragraph packing lost text in {document}/{chunk}")
            for index, text in enumerate(paragraphs):
                yield {
                    "id": f"loogle:{document[:16]}:{chunk}:p{index}", "ordinal": ordinal,
                    "text": text, "images": [],
                    "locator": {"dataset": "loogle", "sample_id": document, "session_index": sequence,
                                "chunk_id": chunk, "dia_id": f"{chunk}.{index}", "speaker": "document"},
                }
                ordinal += 1


def write_rows(path, rows):
    count = 0
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", type=Path, nargs="+", required=True)
    parser.add_argument("--split", type=Path, default=ROOT / "configs/loogle.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gold-output", type=Path, required=True)
    args = parser.parse_args()

    if args.output.exists() or args.gold_output.exists():
        parser.error("use new output paths")
    contexts, by_doc, gold = load_release(args.parquet)
    split = json.loads(args.split.read_text(encoding="utf-8"))
    train, test = set(split["train"]), set(split["test"])
    if train & test or set(contexts) != train | test or len(gold) != 1934:
        parser.error("input documents and question count differ from the reference release")
    if (sum(len(by_doc[d]) for d in train), sum(len(by_doc[d]) for d in test)) != (770, 1164):
        parser.error("input questions differ from the reference split counts")

    args.output.mkdir(parents=True)
    count = write_rows(args.output / "source_records.jsonl", source_records(contexts))
    write_rows(args.output / "public_tasks.jsonl", public_tasks(by_doc))
    args.gold_output.parent.mkdir(parents=True, exist_ok=True)
    write_rows(args.gold_output, gold)
    manifest = {"dataset": "loogle_v2", "documents": len(contexts), "questions": len(gold),
                "source_records": count, "parquet_sha256": {p.name: sha256(p) for p in args.parquet}}
    (args.output / "cell_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
