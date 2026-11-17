"""End-to-end MEMORIUM task-conditioned memory workflow.

The default components are deterministic fallbacks. Model-backed analyzers,
selectors, readers, and image generators plug in through the same callables.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import re
from collections.abc import Mapping
from typing import Any, Callable, Iterable, Protocol

from .core import (
    Candidate,
    CandidateSelection,
    CandidateSet,
    FeedbackRecord,
    Memory,
    MemoryNode,
    MemoryPatch,
    PatchReceipt,
    RetrievalResult,
    TaskSpec,
    Workspace,
    merge_candidates,
    select_candidates,
)


TaskAnalyzer = Callable[[TaskSpec], TaskSpec]
Reader = Callable[[TaskSpec, Workspace], Any]
ImageGenerator = Callable[[TaskSpec, Workspace], Iterable[str]]


class Selector(Protocol):
    def __call__(self, task: TaskSpec, candidates: CandidateSet, memory: Memory) -> Iterable[Candidate]:
        ...


class Generator(Protocol):
    def __call__(self, task: TaskSpec, workspace: Workspace) -> MemoryNode:
        ...


@dataclass(frozen=True)
class TaskRun:
    task: TaskSpec
    retrieval: RetrievalResult
    workspace: Workspace
    generated_node: MemoryNode
    patch_receipt: PatchReceipt
    reader_output: Any = None


def _validate_analysis(original: TaskSpec, analyzed: TaskSpec) -> TaskSpec:
    if not isinstance(analyzed, TaskSpec):
        raise TypeError("task analyzer must return TaskSpec")
    if analyzed.question != original.question or analyzed.options != original.options:
        raise ValueError("task analyzer cannot change the public question or options")
    return analyzed


def _question_intent(question: str) -> str:
    lowered = question.strip().lower()
    if lowered.startswith(("who", "which person", "which player")):
        return "identify the requested entity"
    if lowered.startswith(("where", "what location")):
        return "locate the requested place"
    if lowered.startswith(("when", "what date", "what time")):
        return "identify the requested time"
    if lowered.startswith(("why", "how")):
        return "explain the requested relation or process"
    return "answer the task using relevant multimodal memory"


def default_task_analyzer(task: TaskSpec) -> TaskSpec:
    """Create a usable task memory request without invoking a model."""
    if task.intent and task.subgoals:
        return task
    subgoals = task.subgoals or ("locate relevant evidence", "verify supporting context")
    return replace(task, intent=task.intent or _question_intent(task.question), subgoals=subgoals)


def _tokens(text: str) -> set[str]:
    return {item for item in re.findall(r"\w+", text.lower()) if len(item) > 1}


class KeywordSelector:
    """Small deterministic noise filter used until a learned selector is attached."""

    def __init__(self, limit: int | None = None) -> None:
        self.limit = limit

    def __call__(self, task: TaskSpec, candidates: CandidateSet, memory: Memory) -> Iterable[Candidate]:
        query_terms = _tokens(" ".join((task.public_query, task.intent, *task.subgoals,
                                         *task.constraints)))
        scored: list[tuple[float, int, Candidate]] = []
        for index, candidate in enumerate(candidates.candidates):
            texts: list[str] = []
            if candidate.node_id is not None:
                node = memory.nodes[candidate.node_id]
                texts.append(node.retrieval_summary)
                texts.extend(str(item.get("text", "")) for item in node.canvas.get("items", ())
                             if isinstance(item, Mapping))
            texts.extend(memory.sources[ref].text for ref in candidate.source_refs)
            score = float(len(query_terms & _tokens(" ".join(texts))))
            scored.append((score, index, candidate))
        if not scored:
            return ()
        if all(score == 0.0 for score, _, _ in scored):
            ordered = [candidate for _, _, candidate in scored]
        else:
            ordered = [candidate for score, _, candidate in sorted(
                scored, key=lambda item: (-item[0], item[1])) if score > 0.0]
        return ordered if self.limit is None else ordered[:self.limit]


class StructuredMemoryGenerator:
    """Package a task workspace into a reusable node without model calls."""

    def __init__(self, image_generator: ImageGenerator | None = None) -> None:
        self.image_generator = image_generator

    def __call__(self, task: TaskSpec, workspace: Workspace) -> MemoryNode:
        if not workspace.sources:
            raise ValueError("cannot generate task memory without source evidence")
        node_id = f"task:{task.signature[:20]}"
        source_refs = workspace.source_refs
        summaries = "; ".join(node.retrieval_summary for node in workspace.nodes)
        summary = f"{task.intent or task.question}: {summaries}".strip(" :")
        items = [
            {"kind": "task", "text": task.intent or task.question,
             "task_signature": task.signature},
            *({"kind": "memory_node", "node_id": node.id,
               "text": node.retrieval_summary, "source_refs": list(node.source_refs)}
              for node in workspace.nodes),
        ]
        assets = tuple(self.image_generator(task, workspace)) if self.image_generator else ()
        return MemoryNode(
            id=node_id,
            retrieval_summary=summary,
            source_refs=source_refs,
            canvas={"type": "task_memory", "items": items,
                    "source_refs": list(source_refs)},
            related=tuple(node.id for node in workspace.nodes if node.id != node_id),
            visual_assets=assets,
            task_signature=task.signature,
        )


def expand_local_candidates(memory: Memory, candidates: CandidateSet,
                            max_hops: int = 1) -> tuple[Candidate, ...]:
    """Expand only immediate PageIndex-like neighbors from retrieved nodes."""
    if max_hops < 0:
        raise ValueError("max_hops must be non-negative")
    frontier = tuple(dict.fromkeys(candidate.node_id for candidate in candidates.candidates
                                   if candidate.node_id is not None))
    visited = set(frontier)
    expanded: list[str] = []
    for _ in range(max_hops):
        next_frontier: list[str] = []
        for node_id in frontier:
            neighbors = (memory.navigate(node_id, "parent") +
                         memory.navigate(node_id, "children") +
                         memory.navigate(node_id, "related"))
            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    next_frontier.append(neighbor)
                    expanded.append(neighbor)
        frontier = tuple(next_frontier)
    return memory.candidates("H", expanded)


class MemoriumPipeline:
    """Run automatic task analysis, retrieval, editing, reading, and write-back."""

    def __init__(self, memory: Memory, analyzer: TaskAnalyzer | None = None,
                 selector: Selector | None = None, generator: Generator | None = None,
                 reader: Reader | None = None) -> None:
        self.memory = memory
        self.analyzer = analyzer or default_task_analyzer
        self.selector = selector or KeywordSelector()
        self.generator = generator or StructuredMemoryGenerator()
        self.reader = reader

    def run(self, task: TaskSpec, rankings: dict[str, Iterable[str]], *, limit: int | None = None,
            max_hops: int = 1, feedback: Iterable[FeedbackRecord] = ()) -> TaskRun:
        analyzed = _validate_analysis(task, self.analyzer(task))
        route_candidates = tuple(
            self.memory.candidates(route, rankings.get(route, ()))
            for route in ("T", "V", "H", "D")
        )
        base = merge_candidates(*route_candidates)
        local = expand_local_candidates(self.memory, base, max_hops=max_hops)
        merged = merge_candidates(base.candidates, local)
        selected = tuple(self.selector(analyzed, merged, self.memory))
        if limit is not None:
            selected = selected[:limit]
        selection = select_candidates(merged, limit=limit, selected=selected)
        retrieved_workspace = self.memory.workspace(analyzed, CandidateSet(selection.selected))
        retrieval = RetrievalResult(merged, selection, retrieved_workspace)
        generated = self.generator(analyzed, retrieved_workspace)
        if not isinstance(generated, MemoryNode):
            raise TypeError("memory generator must return MemoryNode")
        if generated.task_signature != analyzed.signature:
            raise ValueError("generated memory node must carry the analyzed task signature")
        if not set(generated.source_refs) <= set(retrieved_workspace.source_refs):
            raise ValueError("generated memory node references source outside the task workspace")
        additions = tuple(feedback)
        if any(ref not in self.memory.sources for item in additions for ref in item.source_refs):
            raise ValueError("feedback references an unknown source")
        if additions:
            generated = replace(generated, feedback=additions, content_hash="")
        receipt = self.memory.apply(MemoryPatch(
            id=f"task:{analyzed.signature}:{self.memory.revision + 1}",
            expected_revision=self.memory.revision,
            upsert=(generated,),
        ))
        persisted = self.memory.nodes[generated.id]
        task_workspace = Workspace(analyzed, retrieved_workspace.nodes + (persisted,),
                                   retrieved_workspace.sources)
        reader_output = self.reader(analyzed, task_workspace) if self.reader else None
        return TaskRun(analyzed, retrieval, task_workspace, persisted, receipt, reader_output)
