"""Build and read a source-linked evidence sheet for an abstained task."""
from __future__ import annotations

import re
from typing import Sequence

from .core import View
from .u05 import (ArmConfig, TaskResult, abstention_phrase, expand, parse_reader, reader_instruction, resolve_citations)
from .unit import text_of
from .views import node_kind
from .experience_keys import entities

_WORD = re.compile(r"[a-z0-9]+")


def read_sheet(unit, task, views: Sequence[View], reader) -> tuple[str, list[str], dict]:
    call = reader(f"{reader_instruction(task)}\n\nQuestion: {task.public_query}", list(views))
    answer, cited, _ = parse_reader(str(call.get("output", "")))
    abstain = abstention_phrase(task)
    if not answer and abstain:
        answer = abstain
    return answer, resolve_citations(unit, cited), call


def edit_sheet(unit, task, first: TaskResult, arm: ArmConfig) -> View | None:
    """Task edit for text memory: select the lines of a wider pool that share the question's words or
    entities, keep them in source order, pack them under the arm's budget. Source lines verbatim only."""
    pool = list(dict.fromkeys(first.retrieved + expand(unit, first.retrieved[:arm.top_k], 1)))
    words = set(_WORD.findall(task.question.lower())) - {"the", "a", "an", "of", "to", "in", "did", "what",
                                                          "when", "who", "is", "was", "do", "does", "how"}
    names = entities(task.question)
    scored, refs = [], []
    for node_id in pool:
        node = unit.memory.nodes[node_id]
        if node_kind(node) not in ("segment", "entity", "timeline", "derived"):
            continue
        for line in text_of(node).splitlines():
            overlap = len(words & set(_WORD.findall(line.lower()))) + 2 * sum(n in line for n in names)
            if overlap:
                scored.append((overlap, line))
    seen, kept, budget = set(), [], arm.token_budget
    for _, line in sorted(scored, key=lambda item: -item[0]):
        if line in seen:
            continue
        cost = unit.config.count_tokens(line)
        if cost > budget:
            continue
        seen.add(line)
        kept.append(line)
        budget -= cost
    if not kept:
        return None
    order = {line: i for i, line in enumerate(text for n in pool for text in text_of(unit.memory.nodes[n]).splitlines())}
    kept.sort(key=lambda line: order.get(line, 0))
    sheet = "Evidence sheet built for this question:\n" + "\n".join(kept)
    refs = tuple(ref for n in pool for ref in unit.memory.nodes[n].source_refs)
    return View("T0", sheet, tuple(dict.fromkeys(refs)), "task_edit_sheet", unit.config.count_tokens(sheet))
