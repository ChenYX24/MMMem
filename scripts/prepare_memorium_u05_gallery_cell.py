#!/usr/bin/env python3
"""Materialize the public, source-only Mem-Gallery cell for MEMORIUM U05; never calls a model.

One source record per dialogue round (user + assistant text), carrying the session date and the round's
real images (paths + sha256) as native image evidence. Public tasks carry only the question, the query
image if any, and the official category-specific answer-format instruction. Gold answers, clues and
session ids stay in the admission source.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

FORBIDDEN = {"answer", "clue", "gold", "label", "rationale", "score", "session_id", "point", "category"}
# Official instructions (benchmark/prompt/{ar,cd,vs}_prompt.txt) keyed by category; others need none.
FORMAT_FILES = {"AR": "ar_prompt.txt", "CD": "cd_prompt.txt", "VS": "vs_prompt.txt"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def key_names(value: Any) -> set[str]:
    if isinstance(value, dict):
        return {str(k).lower() for k in value} | set().union(*(key_names(v) for v in value.values()))
    if isinstance(value, list):
        return set().union(*(key_names(v) for v in value)) if value else set()
    return set()


def resolve_image(relative: str, image_root: Path) -> Path:
    """Admission paths look like '../image/<scenario>/<file>'; resolve against the release image dir."""
    parts = Path(relative).parts
    if "image" not in parts:
        raise ValueError(f"unexpected image path: {relative}")
    return image_root.joinpath(*parts[parts.index("image") + 1:])


def source_rows(writer_rows: list[dict], image_root: Path) -> list[dict]:
    rows, ordinal = [], 0
    for scenario in writer_rows:
        unit = scenario["scenario"]
        speaker = scenario["speaker_name"]
        for session in scenario["sessions"]:
            for turn in session["dialogues"]:
                images, captions = [], []
                for relative, image_id, caption in zip(turn.get("input_image") or [], turn.get("image_id") or [],
                                                       turn.get("image_caption") or []):
                    path = resolve_image(relative, image_root)
                    if not path.is_file():
                        raise FileNotFoundError(path)
                    images.append(str(path))
                    captions.append(f"[{image_id}] {caption}")
                text = f"{speaker}: {turn['user']}\nAssistant: {turn['assistant']}"
                rows.append({
                    "id": f"memgallery:{unit}:{turn['round']}", "ordinal": ordinal, "text": text, "images": images,
                    "image_sha256": [sha256(Path(p)) for p in images],
                    "locator": {"dataset": "memgallery", "sample_id": unit, "session_index": session["session_id"],
                                "dia_id": turn["round"], "speaker": speaker, "date_time": session["date"],
                                "image_ids": list(turn.get("image_id") or []),
                                "visual_caption": " ".join(captions) or None,
                                "visual_representation": "native_image" if images else "none"},
                })
                ordinal += 1
    return rows


def public_tasks(question_rows: list[dict], image_root: Path, formats: dict[str, str]) -> list[dict]:
    tasks, counts = [], {}
    for row in question_rows:
        qa = row["official_qa"]
        unit = row["scenario"]
        counts[unit] = counts.get(unit, 0) + 1
        query_image = qa.get("question_image")
        path = resolve_image(query_image, image_root) if query_image else None
        if path is not None and not path.is_file():
            raise FileNotFoundError(path)
        tasks.append({
            "id": row["id"], "sample_id": unit, "question_index": counts[unit] - 1,
            "question": qa["question"], "options": [],
            "format_instruction": formats.get(qa["point"], ""),
            "query_image": str(path) if path else None,
            "query_image_sha256": sha256(path) if path else None,
        })
    return tasks


def split_source(raw, scenario):
    """Whitelist history fields; question images and QA metadata stay separate."""
    source = {'scenario': scenario, 'speaker_name': raw['character_profile'].get('name', ''), 'sessions': []}
    keys = ('round', 'user', 'assistant', 'input_image', 'image_id', 'image_caption')
    for session in raw['multi_session_dialogues']:
        source['sessions'].append({'session_id': session['session_id'], 'date': session['date'],
                                   'dialogues': [{k: d[k] for k in keys if k in d} for d in session['dialogues']]})
    qa = [{'id': f'memgallery:{scenario}:{i:04d}', 'scenario': scenario, 'official_qa': q}
          for i, q in enumerate(raw['human-annotated QAs'])]
    return source, qa



def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True, help="dataset's data directory")
    parser.add_argument("--prompts", type=Path, required=True, help="benchmark/prompt in the code checkout")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gold-output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.gold_output.exists():
        parser.error("use new output paths")
    image_root = args.release.resolve() / "image"
    formats = {point: (args.prompts / name).read_text(encoding="utf-8").strip() for point, name in FORMAT_FILES.items()}
    system = (args.prompts / "sys_prompt.txt").read_text(encoding="utf-8").strip()
    writer, questions, input_hashes = [], [], {}
    for path in sorted((args.release / "dialog").glob("*.json")):
        source, qa = split_source(json.loads(path.read_text(encoding="utf-8")), path.stem)
        writer.append(source)
        questions.extend(qa)
        input_hashes[path.name] = sha256(path)
    sources = source_rows(writer, image_root)
    tasks = public_tasks(questions, image_root, formats)
    for task in tasks:
        leaked = key_names({k: v for k, v in task.items() if k not in ("format_instruction",)}) & FORBIDDEN
        if leaked:
            raise ValueError(f"{task['id']}: forbidden keys {sorted(leaked)}")
    if len(tasks) != 1711 or len({s['locator']['sample_id'] for s in sources}) != 20:
        raise ValueError(f"unexpected coverage: tasks={len(tasks)}")

    args.output.mkdir(parents=True)
    for name, rows in (("source_records.jsonl", sources), ("public_tasks.jsonl", tasks)):
        (args.output / name).write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows),
                                        encoding="utf-8")
    args.gold_output.parent.mkdir(parents=True, exist_ok=True)
    args.gold_output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in questions),
                                encoding="utf-8")
    manifest = {
        "schema": "memorium-u05-gallery-cell-v1", "dataset": "memgallery", "public_only": True, "source_only": True,
        "counts": {"scenarios": 20, "rounds": len(sources), "tasks": len(tasks),
                   "rounds_with_images": sum(1 for s in sources if s["images"]),
                   "images": sum(len(s["images"]) for s in sources),
                   "tasks_with_query_image": sum(1 for t in tasks if t["query_image"])},
        "system_prompt": system,
        "format_instruction_sources": {p: str(args.prompts / n) for p, n in FORMAT_FILES.items()},
        "boundary": "answers, clues, session ids and categories stay in the admission source; the official "
                    "per-category answer-format instruction is public task input (the benchmark shows it to models)",
        "input_sha256": input_hashes,
    }
    (args.output / "cell_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
                                                   encoding="utf-8")
    print(json.dumps(manifest["counts"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
