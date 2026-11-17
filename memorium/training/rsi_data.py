"""KBAlign-style self-generated QA from train-unit source text only.

The model reads one segment (plus its session date) and proposes questions with an answer and the
record ids that support it. Each item must survive: evidence ids exist in the segment, the answer is
supported by the cited records (string / number / date overlap), and a second independent reading
of the cited records alone reproduces the answer. Official QA of train units is never read here.
"""
from __future__ import annotations

import json
import re
from typing import Callable, Iterable, Sequence

QUESTION_TYPES = ("single_hop", "temporal", "multi_hop", "no_information")

GEN_PROMPT = (
    "Below are conversation records, each starting with [id] speaker (date). Write {n} question-answer "
    "pairs a person could later ask about this conversation. Mix these types: single_hop (one fact), "
    "temporal (when something happened; answer with an absolute date or month), multi_hop (needs two "
    "records), no_information (plausible question the records do NOT answer; answer exactly "
    "'No information available' and give a wrong but plausible alternative). Answers must be short. "
    "Return JSON: [{{\"type\": str, \"question\": str, \"answer\": str, \"evidence\": [ids], "
    "\"distractor\": str (only for no_information)}}]. Return JSON only.\n\nRecords:\n{records}"
)

CHECK_PROMPT = (
    "Answer the question using only these records. Reply with the short answer only, or 'No information "
    "available' if they do not say.\n\nRecords:\n{records}\n\nQuestion: {question}"
)

_ID = re.compile(r"^\[([^\]]+)\]")
_WORDS = re.compile(r"[a-z0-9]+")
_STOP = {"the", "a", "an", "of", "in", "on", "at", "to", "and", "is", "was", "his", "her", "their", "for"}


def record_ids(lines: Sequence[str]) -> dict[str, str]:
    return {match.group(1): line for line in lines if (match := _ID.match(line))}


def content_words(text: str) -> set[str]:
    return set(_WORDS.findall(text.lower())) - _STOP


def supported(answer: str, cited: Iterable[str]) -> bool:
    """At least half of the answer's content words (and every number) appear in the cited records."""
    words = content_words(answer)
    if not words:
        return False
    text = " ".join(cited).lower()
    haystack = content_words(text)
    numbers = {w for w in words if w.isdigit()}
    return numbers <= haystack and len(words & haystack) >= max(1, (len(words) + 1) // 2)


def agrees(first: str, second: str) -> bool:
    a, b = content_words(first), content_words(second)
    if not a or not b:
        return False
    return len(a & b) / len(a | b) >= 0.5 or a <= b or b <= a


def parse_items(raw: str) -> list[dict]:
    match = re.search(r"\[.*\]", raw, re.S)
    if not match:
        return []
    try:
        items = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    return [item for item in items if isinstance(item, dict)]


def generate_for_segment(lines: Sequence[str], generate: Callable[[str], str], check: Callable[[str], str],
                         *, n: int = 4, unit: str = "", segment: str = "") -> tuple[list[dict], dict]:
    """Returns kept items and a reason tally; `generate`/`check` are model calls returning text."""
    ids = record_ids(lines)
    tally = {"proposed": 0, "bad_type": 0, "bad_evidence": 0, "unsupported": 0, "inconsistent": 0, "kept": 0}
    kept = []
    seen_questions: set[str] = set()
    for item in parse_items(generate(GEN_PROMPT.format(n=n, records="\n".join(lines)))):
        tally["proposed"] += 1
        kind = str(item.get("type", "")).strip()
        question, answer = str(item.get("question", "")).strip(), str(item.get("answer", "")).strip()
        evidence = [str(e).strip().strip("[]") for e in item.get("evidence", []) or []]
        if kind not in QUESTION_TYPES or not question or not answer:
            tally["bad_type"] += 1
            continue
        key = " ".join(sorted(content_words(question)))
        if key in seen_questions:
            tally["duplicate"] = tally.get("duplicate", 0) + 1
            continue
        seen_questions.add(key)
        if kind == "no_information":
            distractor = str(item.get("distractor", "")).strip()
            if answer.lower() != "no information available" or not distractor:
                tally["bad_type"] += 1
                continue
            second = check(CHECK_PROMPT.format(records="\n".join(lines), question=question))
            if "no information" not in second.lower():
                tally["inconsistent"] += 1
                continue
            kept.append({"unit": unit, "segment": segment, "type": kind, "question": question,
                         "answer": answer, "evidence": [], "options": sorted([answer, distractor])})
            tally["kept"] += 1
            continue
        if not evidence or any(e not in ids for e in evidence):
            tally["bad_evidence"] += 1
            continue
        cited = [ids[e] for e in evidence]
        if not supported(answer, cited):
            tally["unsupported"] += 1
            continue
        second = check(CHECK_PROMPT.format(records="\n".join(cited), question=question))
        if not agrees(answer, second):
            tally["inconsistent"] += 1
            continue
        kept.append({"unit": unit, "segment": segment, "type": kind, "question": question, "answer": answer,
                     "evidence": evidence})
        tally["kept"] += 1
    return kept, tally
