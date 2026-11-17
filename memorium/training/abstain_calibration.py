"""Abstention forms a domain's own questions use, for calibrating the Reader's training data (amend8).

Read from the public task text of GEN units only (question side, never answers), the same scope as few-shot:
  stated   a format instruction that names the reply to give when the material does not say
           (Mem-Gallery: 'Only if ... is not present in the conversation, reply with: "Not mentioned."')
  option   an answer option that means "cannot tell" (OVO-Bench: "Unable to answer")
Domains whose questions use neither form keep the RSI2_J recipe unchanged.
"""
from __future__ import annotations

from collections import Counter
import random
import re
from typing import Iterable, Mapping, Sequence

from memorium.core import TaskSpec
from memorium.u05 import abstention_phrase

LETTERS = "ABCDE"
ABSTAIN_CHOICES = ("unable to answer", "not mentioned", "no information available")
_OPTION = re.compile(r"^\(([A-E])\)\s*(.+?)\s*$")


def stated_abstention(tasks: Iterable[Mapping]) -> tuple[str, str] | None:
    """(instruction, phrase): the most common format instruction that states an abstention reply."""
    found = Counter()
    for task in tasks:
        instruction = (task.get("format_instruction") or "").strip()
        if instruction:
            phrase = abstention_phrase(TaskSpec(task.get("question", ""), constraints=(instruction,)))
            if phrase:
                found[(instruction, phrase)] += 1
    return found.most_common(1)[0][0] if found else None


def split_choice(text: str) -> tuple[str, list[str]]:
    """'stem\\n(A) x\\n(B) y' -> ('stem', ['x', 'y']); options must be lettered A, B, ... in order."""
    lines = text.rstrip().split("\n")
    options = []
    while lines and _OPTION.match(lines[-1]):
        options.insert(0, lines.pop())
    parsed = [_OPTION.match(line).groups() for line in options]
    if [letter for letter, _ in parsed] != list(LETTERS[:len(parsed)]):
        return text, []
    return "\n".join(lines), [option for _, option in parsed]


def abstention_option(tasks: Iterable[Mapping]) -> str | None:
    """The most common option text meaning 'cannot tell' among the questions' lettered options."""
    found = Counter()
    for task in tasks:
        for option in split_choice(task.get("question", ""))[1]:
            if option.strip().rstrip(".").lower() in ABSTAIN_CHOICES:
                found[option.strip()] += 1
    return found.most_common(1)[0][0] if found else None


def join_choice(stem: str, options: Sequence[str]) -> str:
    return stem.strip() + "\n" + "\n".join(f"({LETTERS[i]}) {o}" for i, o in enumerate(options))


def with_abstention_option(text: str, answer: str, option: str, as_answer: bool,
                           rng: random.Random) -> str | None:
    """Put `option` among a four-option question: in place of the correct option (it becomes the answer, the
    letter is unchanged) or of one wrong option (the answer is unchanged). None if the question does not parse."""
    stem, options = split_choice(text)
    if len(options) != 4 or answer not in LETTERS[:4] or any(
            o.strip().rstrip(".").lower() in ABSTAIN_CHOICES for o in options):
        return None
    correct = LETTERS.index(answer)
    slot = correct if as_answer else rng.choice([i for i in range(4) if i != correct])
    options[slot] = option
    return join_choice(stem, options)
