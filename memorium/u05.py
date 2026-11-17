"""The U05 per-task loop: retrieve over views, select nodes x views, read, write back, upgrade.

Model-backed pieces (encoders, reranker, reader, writers, image generator) are injected, so the
whole loop runs on CPU with deterministic stand-ins for tests and dry runs.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import re
from typing import Any, Callable, Sequence

from .core import MemoryNode, TaskSpec, View
from .unit import (BuiltUnit, attach_view, image_items, link_note_to_entities, retrieval_texts, text_of,
                   write_note)
from .views import MaterializationPolicy, node_kind

Reader = Callable[[str, Sequence[str]], dict]
Reranker = Callable[[str, Sequence[str]], Sequence[float]]

READER_INSTRUCTION = (
    "Answer the question using only the memory below. Each record starts with [id] speaker (date). "
    "Memory notes marked 'unverified' are your own earlier answers; trust the records over them.\n"
    "Reply in exactly this format:\n"
    "Answer: <short answer>\n"
    "Evidence: <comma-separated record ids you used>\n"
    "If the memory does not contain enough information, reply instead:\n"
    "Answer: insufficient\n"
    "Need: <what to look up next>"
)
OPTIONS_INSTRUCTION = "The question lists options (a), (b), ...: answer with the text of exactly one option."
ABSTAIN_OPTIONS = ("no information available", "not mentioned")


def abstention_option(options: Sequence[str]) -> str | None:
    """The option that means 'the memory does not say', if the task offers one."""
    return next((o for o in options if o.strip().lower() in ABSTAIN_OPTIONS), None)


_QUOTED = re.compile(r"[\u201c\"]([^\u201d\"]+)[\u201d\"]")


def abstention_phrase(task: TaskSpec) -> str | None:
    """A task's own stated abstention reply (e.g. the benchmark says to answer "Not mentioned.")."""
    option = abstention_option(task.options)
    if option:
        return option
    for constraint in task.constraints:
        for phrase in _QUOTED.findall(constraint):
            if phrase.strip().lower().rstrip(".") in {a.rstrip(".") for a in ABSTAIN_OPTIONS}:
                return phrase.strip()
    return None


@dataclass
class ArmConfig:
    """One experimental arm; each switch maps to one row of the pre-registered ablation."""

    routes: tuple[str, ...] = ("T", "V")
    write_back: bool = True
    iterative: bool = False
    reader_memory: str = "text"
    token_budget: int = 3000
    top_k: int = 12
    hops: int = 1
    # Write-back controls (U05.1). Defaults reproduce the U05 A1 behaviour exactly.
    note_pull: str = "segments"
    note_limit: int | None = None
    note_min_sim: float | None = None
    note_style: str = "answer"
    # Task views (memorium/editing.py) are written back for reuse by later plans; whether general retrieval
    # may also return them is a separate switch.
    taskview_retrieval: bool = False
    # Retain the question-picture route when a text-only reranker is used.
    preserve_query_image_rank: bool = False
    # Optionally pass generated scene views to the reader within the evidence budget.
    read_generated_views: bool = False


def build_indices(unit: BuiltUnit, text_index, visual_index) -> None:
    """Index every node's views; T gets text views, V gets image views, scoped to this unit."""
    text_entries, image_entries = [], []
    for node_id, node in unit.memory.nodes.items():
        if node_kind(node) == "session":
            # Sessions carry no text of their own; a session-level page (long documents) is a V entry.
            image_entries += [(node_id, unit.unit_id, item) for item in image_items(node)]
            continue
        text_entries += [(node_id, unit.unit_id, item) for item in retrieval_texts(node)]
        image_entries += [(node_id, unit.unit_id, item) for item in image_items(node)]
    text_index.add(text_entries)
    if visual_index is not None:
        visual_index.add(image_entries)


def index_node(unit: BuiltUnit, node_id: str, text_index, visual_index) -> None:
    node = unit.memory.nodes[node_id]
    text_index.add([(node_id, unit.unit_id, item) for item in retrieval_texts(node)])
    if visual_index is not None:
        visual_index.add([(node_id, unit.unit_id, item) for item in image_items(node)])


def rrf(rankings: Sequence[Sequence[str]], k: int = 60) -> list[str]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, node in enumerate(ranking, 1):
            scores[node] = scores.get(node, 0.0) + 1.0 / (k + rank)
    return [node for node, _ in sorted(scores.items(), key=lambda item: (-item[1], item[0]))]


def expand(unit: BuiltUnit, seeds: Sequence[str], hops: int) -> list[str]:
    """Structural neighbours: related links (entity/timeline/note -> segments) and sibling segments."""
    seen, frontier, out = set(seeds), list(seeds), []
    for _ in range(hops):
        nxt = []
        for node_id in frontier:
            node = unit.memory.nodes[node_id]
            neighbours = list(node.related)
            if node_kind(node) == "session":
                neighbours += list(node.canvas.get("children", ()))
            if node_kind(node) == "segment" and node.parent:
                siblings = unit.memory.nodes[node.parent].canvas.get("children", ())
                if node_id in siblings:
                    position = list(siblings).index(node_id)
                    neighbours += [siblings[i] for i in (position - 1, position + 1) if 0 <= i < len(siblings)]
            for neighbour in neighbours:
                if neighbour not in seen and neighbour in unit.memory.nodes:
                    seen.add(neighbour)
                    nxt.append(neighbour)
                    out.append(neighbour)
        frontier = nxt
    return out


def choose_view(node: MemoryNode, remaining: int, reader_memory: str) -> View | None:
    """Cheapest view that fits; the image-only arm prefers images, otherwise text is exact."""
    views = list(node.views)
    if reader_memory == "image":
        order = [v for v in views if v.modality == "image"] + [v for v in views if v.modality == "text"]
    else:
        order = sorted((v for v in views if v.modality == "text"), key=lambda v: (v.kind != "T1", v.token_cost))
        order += [v for v in views if v.kind == "V0"]
    for view in order:
        if view.token_cost <= remaining:
            return view
    return None


@dataclass
class Selection:
    nodes: list[str] = field(default_factory=list)
    views: list[View] = field(default_factory=list)
    text_tokens: int = 0
    image_tokens: int = 0
    view_nodes: list[str] = field(default_factory=list)


def select(unit: BuiltUnit, question: str, candidates: Sequence[str], arm: ArmConfig,
           reranker: Reranker | None, image_ranking: Sequence[str] = ()) -> Selection:
    if reranker is not None and candidates:
        scores = reranker(question, [text_of(unit.memory.nodes[c]) for c in candidates])
        candidates = [c for _, c in sorted(zip(scores, candidates), key=lambda item: (-item[0], candidates.index(item[1])))]
        if image_ranking:
            # Fuse two ranks, not incomparable visual cosine and cross-encoder logits. The visual
            # route can reorder this same candidate pool; it cannot introduce extra evidence.
            allowed = set(candidates)
            candidates = rrf([candidates, list(dict.fromkeys(n for n in image_ranking if n in allowed))])
    # A note is the model's own unverified claim: its cited evidence follows it so the reader can check it.
    ordered: list[str] = []
    notes = 0
    for node_id in candidates:
        is_note = node_kind(unit.memory.nodes[node_id]) == "note"
        if is_note:
            if arm.note_limit is not None and notes >= arm.note_limit:
                continue
            notes += 1
        if node_id not in ordered:
            ordered.append(node_id)
        if is_note and arm.note_pull == "segments":
            ordered += [n for n in unit.memory.nodes[node_id].related if n not in ordered]
    candidates = ordered
    chosen = Selection()
    remaining = arm.token_budget
    for node_id in candidates:
        node = unit.memory.nodes[node_id]
        view = choose_view(node, remaining, arm.reader_memory)
        if view is None:
            continue
        chosen.nodes.append(node_id)
        # A native photo is content, not a rendering of the text: it travels with its node in every
        # reader form while budget allows. Nodes without V0 views (all of LoCoMo) are unaffected. A task
        # view (V3, written back by task-driven editing) travels with its node the same way.
        companions = [v for v in node.views if v.kind in ("V0", "V3") and v is not view]
        if arm.read_generated_views:
            companions += [v for v in node.views if v.kind == "V2" and v is not view]
        for item in [view] + companions:
            if item is not view and item.token_cost > remaining:
                continue
            chosen.views.append(item)
            chosen.view_nodes.append(node_id)
            remaining -= item.token_cost
            if item.modality == "image":
                chosen.image_tokens += item.token_cost
            else:
                chosen.text_tokens += item.token_cost
    return chosen


def reader_instruction(task: TaskSpec) -> str:
    instruction = READER_INSTRUCTION + (f"\n{OPTIONS_INSTRUCTION}" if task.options else "")
    if task.constraints:
        instruction += "\n" + "\n".join(task.constraints)
    return instruction


# [ \t]* rather than \s*: a blank "Answer:" line must not swallow the next line.
_ANSWER = re.compile(r"^[ \t]*Answer:[ \t]*(.*)$", re.I | re.M)
_EVIDENCE = re.compile(r"^[ \t]*Evidence:[ \t]*(.*)$", re.I | re.M)
_NEED = re.compile(r"^[ \t]*Need:[ \t]*(.*)$", re.I | re.M)


def parse_reader(raw: str) -> tuple[str, list[str], str | None]:
    answer = _ANSWER.search(raw)
    text = (answer.group(1) if answer else raw.strip().splitlines()[0] if raw.strip() else "").strip()
    evidence = _EVIDENCE.search(raw)
    cited = [item.strip().strip("[]") for item in evidence.group(1).split(",")] if evidence else []
    need = _NEED.search(raw)
    insufficient = text.lower().startswith("insufficient")
    return ("" if insufficient else text), [c for c in cited if c], (need.group(1).strip() if need and insufficient else None)


def resolve_citations(unit: BuiltUnit, cited: Sequence[str]) -> list[str]:
    """Map short ids like 'D1:3' back to full source ids within this unit."""
    lookup = {}
    for source_id, source in unit.memory.sources.items():
        lookup[source_id] = source_id
        dia = source.locator.get("dia_id")
        if dia:
            lookup[str(dia)] = source_id
    return [lookup[c] for c in cited if c in lookup]


@dataclass
class TaskResult:
    task_id: str
    answer: str
    raw: str
    retrieved: list[str]
    selected: list[str]
    views: list[str]
    text_tokens: int
    image_tokens: int
    iterations: int
    note: str | None
    note_updated: bool
    upgrades: list[str]
    cited: list[str]
    reader_calls: list[dict]
    budget_halvings: int = 0
    # Experience admission log (staging.s.admission on): ranked candidates and the decision on the top-1 node.
    experience_admission: dict | None = None


def run_task(unit: BuiltUnit, task: TaskSpec, task_id: str, arm: ArmConfig, *, text_index, visual_index,
             reader: Reader, reranker: Reranker | None = None, policy: MaterializationPolicy | None = None,
             upgrade: Callable[[BuiltUnit, str, str], View | None] | None = None,
             query_image: str | None = None, attach: View | None = None,
             attach_note: str = "", staging=None, step: int = 0,
             prefer: Sequence[str] = (), query_image_sha256: str | None = None) -> TaskResult:
    """attach: an extra image view (a task view made for this question) shown after the selected evidence; its
    cost comes out of the token budget first, and the node that holds it is not selected a second time.
    query_image_sha256: identity of the question's picture for experience admission (computed from query_image
    when not given); used only when the experience store has admission on."""
    query = task.public_query
    if attach is not None:
        arm = replace(arm, token_budget=max(0, arm.token_budget - attach.token_cost))
    calls: list[dict] = []
    retrieved: list[str] = []
    selection = Selection()
    raw = answer = ""
    cited: list[str] = []
    iterations = 0
    halvings = 0
    admission_log = None
    admitting = staging is not None and getattr(getattr(staging, "s", None), "admission", False)
    if admitting and query_image is not None and query_image_sha256 is None:
        from .experience import image_sha256
        query_image_sha256 = image_sha256(query_image)
    for iterations in range(1, (2 if arm.iterative else 1) + 1):
        rankings = []
        image_ranking = []
        if "T" in arm.routes:
            rankings.append(text_index.rank(query, unit.unit_id, top_k=arm.top_k))
        if "V" in arm.routes and visual_index is not None:
            rankings.append(visual_index.rank(query, unit.unit_id, top_k=arm.top_k))
            if query_image is not None:
                # The question shows a picture: search image views with the picture itself.
                image_ranking = visual_index.rank_item(query_image, unit.unit_id, top_k=arm.top_k)
                rankings.append(image_ranking)
        similarity = (text_index.scores(query, unit.unit_id,
                                        allowed=lambda n: node_kind(unit.memory.nodes[n]) == "note")
                      if arm.write_back and arm.note_min_sim is not None else None)

        def admitted(node: str) -> bool:
            # Applied to direct hits and to expansion alike: notes also arrive via entity links.
            if attach is not None and any(v.content == attach.content for v in unit.memory.nodes[node].views):
                return False
            if node_kind(unit.memory.nodes[node]) == "taskview" and not arm.taskview_retrieval:
                return False
            if node_kind(unit.memory.nodes[node]) != "note":
                return True
            if not arm.write_back:
                return False
            return similarity is None or similarity.get(node, -1.0) >= arm.note_min_sim

        outside = staging is not None and getattr(getattr(staging, "s", None), "outside", False)
        if outside:
            # Experience nodes stay out of the budgeted ranking; the best one is appended after selection.
            _admitted = admitted
            admitted = lambda n: node_kind(unit.memory.nodes[n]) != "experience" and _admitted(n)
            free_hits = staging.rank(query, unit.unit_id, step, lambda n: n in unit.memory.nodes and _admitted(n))
        elif staging is not None:
            # Staged keys point at source nodes; the staged entry itself never reaches the reader.
            rankings.append(staging.rank(query, unit.unit_id, step,
                                         lambda n: n in unit.memory.nodes and admitted(n)))
        base = [node for node in rrf(rankings) if admitted(node)]
        if prefer:
            # Staged-first retrieval: the workspace's source pointers lead, then the ordinary ranking.
            base = [n for n in prefer if n in unit.memory.nodes and admitted(n)] + [n for n in base if n not in prefer]
        pool = base + [node for node in expand(unit, base, arm.hops) if node not in base and admitted(node)]
        retrieved = list(dict.fromkeys(retrieved + pool))
        free_views = []
        if outside and free_hits:
            if admitting:
                # Pre-reader admission: after the priority ranking, before the reader (memorium/experience.py).
                decision, free_views = staging.admit(free_hits[0], query_image_sha256)
                admission_log = {"iteration": iterations, "candidates": staging.candidates(free_hits),
                                 "admission": decision}
            else:
                free_views = [v for v in unit.memory.nodes[free_hits[0]].views if v.kind in ("T0", "V3")]
        while True:
            # The token count used for the budget is an estimate; on dense text (hashes, base64) the real count can
            # be several times larger. A call that overflows the model context is retried with half the evidence
            # budget, and every halving is reported (TaskResult.budget_halvings). Calls that fit are untouched.
            sel_arm = arm if not halvings else replace(arm, token_budget=max(1, arm.token_budget >> halvings))
            selection = select(unit, task.question, retrieved, sel_arm, reranker,
                               image_ranking if arm.preserve_query_image_rank else ())
            if outside and free_hits and (free_views or not admitting):  # admission "skip": nothing is sent
                node_id = free_hits[0]
                for view in free_views:
                    selection.views.append(view)
                    selection.view_nodes.append(node_id)
                    if view.modality == "image":
                        selection.image_tokens += view.token_cost
                    else:
                        selection.text_tokens += view.token_cost
                selection.nodes.append(node_id)
            if attach is not None:
                selection.views.append(attach)
                selection.view_nodes.append(f"task:{task_id}")
                selection.image_tokens += attach.token_cost
            instruction = reader_instruction(task)
            if attach is not None and attach_note:
                instruction += "\n" + attach_note
            try:
                call = reader(f"{instruction}\n\nQuestion: {task.public_query}", selection.views)
                break
            except ValueError as exc:
                if "context cap exceeded" not in str(exc) or sel_arm.token_budget <= 250:
                    raise
                halvings += 1
        calls.append(call)
        raw = str(call.get("output", ""))
        answer, cited, need = parse_reader(raw)
        abstain = abstention_phrase(task)
        if not answer and abstain and (not arm.iterative or iterations == 2 or not need):
            # Offered an explicit "no information" choice, "insufficient" is that choice, not a blank.
            answer, need = abstain, None
        if answer or not arm.iterative or not need:
            break
        query = f"{task.question} {need}"

    note, note_updated, upgrades = None, False, []
    if arm.write_back and answer and answer != abstention_phrase(task) and selection.nodes:
        refs = resolve_citations(unit, cited)
        note, note_updated = write_note(unit.memory, unit.unit_id, task.question, answer, refs, selection.nodes,
                                        count_tokens=unit.config.count_tokens, style=arm.note_style,
                                        normalize_time=unit.config.normalize_time)
        link_note_to_entities(unit.memory, note)
        index_node(unit, note, text_index, visual_index)
    if policy is not None and upgrade is not None:
        for node_id in selection.nodes:
            unit.ledger.use(node_id)
            for kind in policy.upgrades(unit.memory.nodes[node_id], unit.ledger):
                view = upgrade(unit, node_id, kind)
                if view is not None:
                    attach_view(unit.memory, node_id, view, f"{node_id}:upgrade:{kind}:{unit.memory.revision + 1}")
                    index_node(unit, node_id, text_index, visual_index)
                    upgrades.append(f"{node_id}:{kind}")
    # An "insufficient" reply is scored as the empty answer it is; never rewritten into an abstention.
    return TaskResult(task_id, answer, raw, retrieved,
                      selection.nodes, [f"{n}:{v.kind}" for n, v in zip(selection.view_nodes, selection.views)],
                      selection.text_tokens, selection.image_tokens, iterations, note, note_updated, upgrades,
                      resolve_citations(unit, cited), calls, halvings, admission_log)
