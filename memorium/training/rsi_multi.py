"""Domain-general self-generated QA for joint Reader training (extends rsi_data, LoCoMo-only before).

Two answer styles, matching how each benchmark is asked at evaluation time:
  free   (LoCoMo, Mem-Gallery): short free-form answers; types single_hop / temporal / multi_hop / no_information
  choice (LooGLE-v2, LVBench, OVO): four options A-D, answer is a letter; the evaluation forbids abstaining
Two recipes:
  independent  one call per segment, items unrelated to each other
  sequence     one call per segment asks for a related task sequence over the same records: a fact, a paraphrase
               of it (same family), a different need on the same records, and (free style) a near miss that swaps
               a person/date so the records no longer answer it. Items carry sequence_id / role / family.
Every kept item cites record ids present in the segment, is supported by the cited text (free style) and is
reproduced by a second, independent reading of the cited records alone. Official QA is never read.
"""
from __future__ import annotations

import random
import re
from typing import Callable, Sequence

from .rsi_data import CHECK_PROMPT, GEN_PROMPT, agrees, content_words, parse_items, record_ids, supported

LETTERS = "ABCD"

FREE_IMAGE_NOTE = ("Some records shared photos; the photos are attached in record order. Questions may ask about "
                   "what a photo shows, citing the record that shared it.\n")
VIDEO_NOTE = ("These records describe consecutive clips of one video; the attached frames are keyframes of those "
              "clips in order. Questions may ask about what is visible, order of events, counts or reasons.\n")

CHOICE_PROMPT = (
    "{note}Below are records, each starting with [id]. Write {n} multiple-choice questions a person could later ask "
    "about this material. Each question has exactly four options; exactly one is correct and the other three are "
    "plausible but contradicted or unsupported by the records. Mix: detail (one fact), order/time, count or "
    "comparison, reason. Return JSON: [{{\"type\": str, \"question\": str, \"options\": [4 strings], "
    "\"answer_index\": 0-3, \"evidence\": [ids]}}]. Return JSON only.\n\nRecords:\n{records}"
)

CHOICE_CHECK = ("Using only these records, answer the multiple-choice question with only the letter (A, B, C or D)."
                "\n\nRecords:\n{records}\n\nQuestion: {question}")

SEQ_FREE_PROMPT = (
    "{note}Below are conversation records, each starting with [id] speaker (date). Write one related task sequence "
    "of four questions a user might ask at different times about the SAME records:\n"
    "1. fact: one fact; 2. paraphrase: the same fact asked differently; 3. other_need: a different fact or a "
    "comparison from these records; 4. near_miss: like question 1 but with a different person or date so the "
    "records do NOT answer it (answer exactly 'No information available' and give a plausible wrong "
    "alternative as distractor).\nReturn JSON: [{{\"step\": \"fact\"|\"paraphrase\"|\"other_need\"|\"near_miss\", \"question\": str, \"answer\": str, "
    "\"evidence\": [ids], \"distractor\": str (near_miss only)}}]. Return JSON only.\n\nRecords:\n{records}"
)

SEQ_CHOICE_PROMPT = (
    "{note}Below are records, each starting with [id]. Write one related sequence of three multiple-choice "
    "questions about the SAME records: 1. fact; 2. paraphrase of the same fact with reordered options; "
    "3. other_need: a different fact, order or count from these records. Four options each, one correct.\n"
    "Return JSON: [{{\"step\": \"fact\"|\"paraphrase\"|\"other_need\", \"question\": str, \"options\": [4 strings], "
    "\"answer_index\": 0-3, \"evidence\": [ids]}}]. Return JSON only.\n\nRecords:\n{records}"
)

SEQ_ROLES = ("fact", "paraphrase", "other_need", "near_miss")


def mc_text(question: str, options: Sequence[str]) -> str:
    return question.strip() + "\n" + "\n".join(f"({LETTERS[i]}) {o}" for i, o in enumerate(options))


def _shuffle_options(options: list[str], correct: int, rng: random.Random) -> tuple[list[str], int]:
    order = list(range(4))
    rng.shuffle(order)
    return [options[i] for i in order], order.index(correct)


def _letter(text: str) -> str | None:
    match = re.search(r"\b([ABCD])\b", text.strip().upper()[:20])
    return match.group(1) if match else None


def _choice_item(item: dict, ids: dict, check, rng, tally) -> dict | None:
    question = str(item.get("question", "")).strip()
    options = [str(o).strip() for o in item.get("options", []) or []]
    evidence = [str(e).strip().strip("[]") for e in item.get("evidence", []) or []]
    try:
        correct = int(item.get("answer_index"))
    except (TypeError, ValueError):
        correct = -1
    if not question or len(options) != 4 or len(set(options)) != 4 or not 0 <= correct < 4 or not all(options):
        tally["bad_type"] += 1
        return None
    if not evidence or any(e not in ids for e in evidence):
        tally["bad_evidence"] += 1
        return None
    options, correct = _shuffle_options(options, correct, rng)
    text = mc_text(question, options)
    second = _letter(check(CHOICE_CHECK.format(records="\n".join(ids[e] for e in evidence), question=text)))
    if second != LETTERS[correct]:
        tally["inconsistent"] += 1
        return None
    return {"question": text, "answer": LETTERS[correct], "evidence": evidence, "style": "choice"}


def _free_item(item: dict, kind: str, lines, ids: dict, check, tally) -> dict | None:
    question, answer = str(item.get("question", "")).strip(), str(item.get("answer", "")).strip()
    evidence = [str(e).strip().strip("[]") for e in item.get("evidence", []) or []]
    if not question or not answer:
        tally["bad_type"] += 1
        return None
    if kind in ("no_information", "near_miss"):
        distractor = str(item.get("distractor", "")).strip()
        if answer.lower() != "no information available" or not distractor:
            tally["bad_type"] += 1
            return None
        if "no information" not in check(CHECK_PROMPT.format(records="\n".join(lines), question=question)).lower():
            tally["inconsistent"] += 1
            return None
        return {"question": question, "answer": answer, "evidence": [], "options": sorted([answer, distractor]),
                "style": "free"}
    if not evidence or any(e not in ids for e in evidence):
        tally["bad_evidence"] += 1
        return None
    cited = [ids[e] for e in evidence]
    if not supported(answer, cited):
        tally["unsupported"] += 1
        return None
    if not agrees(answer, check(CHECK_PROMPT.format(records="\n".join(cited), question=question))):
        tally["inconsistent"] += 1
        return None
    return {"question": question, "answer": answer, "evidence": evidence, "style": "free"}


FEWSHOT_NOTE = ("Here are real questions users asked about other material of this kind. Match their style, "
                 "difficulty and mix of question types, but write new questions about the records below; never "
                 "copy these:\n{examples}\n\n")


def generate(lines: Sequence[str], images: Sequence[str], style: str, recipe: str,
             call: Callable[[str, Sequence[str], int], str], *, n: int, rng: random.Random,
             note: str = "", unit: str = "", segment: str = "",
             fewshot: Sequence[str] = (), round: int = 1) -> tuple[list[dict], dict]:
    """`call(prompt, images, max_new_tokens)` -> text. Returns kept items and a reason tally.
    From round 2 on the sequence id carries the round, so a trainer pooling rounds never merges two streams."""
    ids = record_ids(lines)
    tally = {"proposed": 0, "bad_type": 0, "bad_evidence": 0, "unsupported": 0, "inconsistent": 0,
             "duplicate": 0, "kept": 0}
    check = lambda prompt: call(prompt, (), 64)
    if recipe == "sequence":
        prompt = (SEQ_CHOICE_PROMPT if style == "choice" else SEQ_FREE_PROMPT).format(note=note, records="\n".join(lines))
    elif style == "choice":
        prompt = CHOICE_PROMPT.format(note=note, n=n, records="\n".join(lines))
    else:
        prompt = note + GEN_PROMPT.format(n=n, records="\n".join(lines))
    if fewshot:
        prompt = FEWSHOT_NOTE.format(examples="\n".join(f"- {q}" for q in fewshot)) + prompt
    raw = call(prompt, images, 1024)
    kept, seen = [], set()
    seq_id = (f"{unit}|{segment}|seq" + (f"|r{round}" if round > 1 else "")) if recipe == "sequence" else None
    for position, item in enumerate(parse_items(raw)):
        tally["proposed"] += 1
        if recipe == "sequence":
            kind = str(item.get("step", "")).strip()
            if kind not in SEQ_ROLES:  # fall back to the requested order
                kind = SEQ_ROLES[min(position, len(SEQ_ROLES) - 1)]
            if style == "free" and str(item.get("answer", "")).strip().lower() == "no information available":
                kind = "near_miss"
        else:
            kind = str(item.get("type", "")).strip() or "fact"
        key = " ".join(sorted(content_words(str(item.get("question", "")))))
        if key in seen:
            tally["duplicate"] += 1
            continue
        seen.add(key)
        made = (_choice_item(item, ids, check, rng, tally) if style == "choice"
                else _free_item(item, kind, lines, ids, check, tally))
        if made is None:
            continue
        made.update(unit=unit, segment=segment, type=kind, recipe=recipe)
        if seq_id:
            # fact and paraphrase share a family; other_need and near_miss are their own families
            made.update(sequence_id=seq_id, position=position,
                        family=f"{seq_id}|{'fact' if kind in ('fact', 'paraphrase') else kind}")
        kept.append(made)
        tally["kept"] += 1
    return kept, tally
