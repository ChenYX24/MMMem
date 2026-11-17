"""Small, provenance-preserving task-conditioned memory contracts."""
from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass, replace
import hashlib
import json
from types import MappingProxyType
from threading import RLock
from typing import Any, Iterable, Mapping


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _canonical(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _canonical(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_canonical(item) for item in value]
    if is_dataclass(value):
        return {item.name: _canonical(getattr(value, item.name)) for item in fields(value)}
    return value


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(_canonical(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


ATTRIBUTION_KEYS = ("speaker", "date_time")


def source_header(source_id: str, locator: Mapping[str, Any]) -> str:
    """Render a source's provenance header, keeping who said it and when."""
    attribution = " @ ".join(str(locator[key]) for key in ATTRIBUTION_KEYS if locator.get(key))
    return f"[{source_id}] ({attribution})" if attribution else f"[{source_id}]"


@dataclass(frozen=True)
class SourceRecord:
    id: str
    ordinal: int
    text: str
    locator: Mapping[str, Any]
    images: tuple[str, ...] = ()
    content_hash: str = ""

    def __post_init__(self) -> None:
        if not self.id or self.ordinal < 0 or not isinstance(self.text, str) or not self.locator:
            raise ValueError("invalid source record")
        object.__setattr__(self, "locator", _freeze(self.locator))
        object.__setattr__(self, "images", tuple(self.images))
        expected = self.recompute_hash()
        if self.content_hash and self.content_hash != expected:
            raise ValueError("source content hash mismatch")
        object.__setattr__(self, "content_hash", expected)

    def recompute_hash(self) -> str:
        return _digest({"id": self.id, "ordinal": self.ordinal, "text": self.text,
                        "locator": dict(self.locator), "images": self.images})

    @property
    def view_header(self) -> str:
        return source_header(self.id, self.locator)

    def source_refs(self) -> tuple[str, ...]:
        return (self.id,)


def _contains_forbidden(value: object) -> bool:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = "".join(char for char in str(key).lower() if char.isalnum())
            if any(token in normalized for token in ("answer", "gold", "score", "label", "rationale")):
                return True
            if _contains_forbidden(item):
                return True
    elif isinstance(value, (list, tuple)):
        return any(_contains_forbidden(item) for item in value)
    return False


@dataclass(frozen=True)
class TaskSpec:
    question: str
    options: tuple[str, ...] = ()
    intent: str = ""
    subgoals: tuple[str, ...] = ()
    constraints: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.question or any(not option for option in self.options):
            raise ValueError("invalid public task")
        if _contains_forbidden(self.metadata):
            raise ValueError("TaskSpec cannot contain answers, gold, labels, rationales, or scores")
        object.__setattr__(self, "options", tuple(self.options))
        object.__setattr__(self, "subgoals", tuple(self.subgoals))
        object.__setattr__(self, "constraints", tuple(self.constraints))
        object.__setattr__(self, "metadata", _freeze(self.metadata))

    @property
    def public_query(self) -> str:
        # Lettered choices: benchmark scorers parse "(a)" as well as the full option text.
        if not self.options:
            return self.question
        choices = "\n".join(f"({chr(ord('a') + index)}) {option}"
                            for index, option in enumerate(self.options))
        return f"{self.question}\n{choices}"

    @property
    def signature(self) -> str:
        """Stable task identity without answers or source content."""
        return _digest({"question": self.question, "options": self.options,
                        "intent": self.intent, "subgoals": self.subgoals,
                        "constraints": self.constraints})


@dataclass(frozen=True)
class Candidate:
    route: str
    rank: int
    source_refs: tuple[str, ...]
    node_id: str | None = None
    node_version: int | None = None

    def __post_init__(self) -> None:
        if (self.route not in {"T", "V", "H", "D"} or self.rank < 1 or not self.source_refs
                or self.node_version is not None and self.node_version < 0
                or len(set(self.source_refs)) != len(self.source_refs)):
            raise ValueError("invalid candidate")


@dataclass(frozen=True)
class CandidateSet:
    candidates: tuple[Candidate, ...]

    @property
    def source_refs(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(ref for candidate in self.candidates for ref in candidate.source_refs))


@dataclass(frozen=True)
class CandidateSelection:
    selected: tuple[Candidate, ...]
    dropped: tuple[Candidate, ...]


@dataclass(frozen=True)
class FeedbackRecord:
    """Optional post-task signal kept separate from public task input."""

    kind: str
    text: str
    source_refs: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in {"correct", "incorrect", "correction", "note"} or not self.text.strip():
            raise ValueError("invalid feedback record")
        object.__setattr__(self, "source_refs", tuple(self.source_refs))
        object.__setattr__(self, "metadata", _freeze(self.metadata))


@dataclass(frozen=True)
class Workspace:
    task: TaskSpec
    nodes: tuple[MemoryNode, ...]
    sources: tuple[SourceRecord, ...]

    @property
    def source_refs(self) -> tuple[str, ...]:
        return tuple(source.id for source in self.sources)

    @property
    def text_view(self) -> str:
        return "\n".join(f"{node.id}: {node.retrieval_summary}" for node in self.nodes)

    @property
    def source_text_view(self) -> str:
        return "\n\n".join(f"{source.view_header}\n{source.text}" for source in self.sources)


@dataclass(frozen=True)
class RetrievalResult:
    candidates: CandidateSet
    selection: CandidateSelection
    workspace: Workspace


@dataclass(frozen=True)
class CostEvent:
    phase: str
    model: str
    call_id: str
    text_tokens: int = 0
    visual_tokens: int = 0
    output_tokens: int = 0
    cpu_seconds: float = 0.0
    gpu_seconds: float = 0.0
    wall_seconds: float = 0.0
    success: bool = True
    cache_status: str = "miss"

    def __post_init__(self) -> None:
        counts = (self.text_tokens, self.visual_tokens, self.output_tokens)
        durations = (self.cpu_seconds, self.gpu_seconds, self.wall_seconds)
        if not self.phase or not self.model or not self.call_id or any(value < 0 for value in counts + durations):
            raise ValueError("invalid cost event")


@dataclass(frozen=True)
class CostTotal:
    text_tokens: int = 0
    visual_tokens: int = 0
    output_tokens: int = 0
    cpu_seconds: float = 0.0
    gpu_seconds: float = 0.0
    wall_seconds: float = 0.0


class CostLedger:
    def __init__(self) -> None:
        self._events: dict[str, CostEvent] = {}

    @property
    def events(self) -> tuple[CostEvent, ...]:
        return tuple(self._events.values())

    @property
    def total(self) -> CostTotal:
        return CostTotal(*(sum(getattr(event, field) for event in self._events.values())
                           for field in CostTotal.__dataclass_fields__))

    def record(self, event: CostEvent) -> bool:
        prior = self._events.get(event.call_id)
        if prior is None:
            self._events[event.call_id] = event
            return True
        if prior != event:
            raise ValueError("conflicting cost event id")
        return False


def merge_candidates(*route_candidates: Iterable[Candidate]) -> CandidateSet:
    """Preserve route order and every distinct source span; budget policy is separate."""
    candidates: list[Candidate] = []
    seen: set[Candidate] = set()
    for route in route_candidates:
        for candidate in route:
            if candidate not in seen:
                candidates.append(candidate)
                seen.add(candidate)
    return CandidateSet(tuple(candidates))


def select_candidates(candidates: CandidateSet, limit: int | None = None,
                      selected: Iterable[Candidate] | None = None) -> CandidateSelection:
    if limit is not None and limit < 0:
        raise ValueError("candidate limit must be non-negative")
    chosen = (candidates.candidates if limit is None else candidates.candidates[:limit]) \
        if selected is None else tuple(selected)
    if (len(chosen) != len(set(chosen)) or any(item not in candidates.candidates for item in chosen)
            or (limit is not None and len(chosen) > limit)):
        raise ValueError("selected candidate is outside the merged union or budget")
    return CandidateSelection(chosen, tuple(item for item in candidates.candidates if item not in chosen))


VIEW_KINDS = {"T0": "text", "T1": "text", "V0": "image", "V1": "image", "V2": "image"}


@dataclass(frozen=True)
class View:
    """One optional representation of a node: raw/compressed text, or native/rendered/generated image."""

    kind: str
    content: str
    source_refs: tuple[str, ...]
    generator: str
    token_cost: int = 0
    verified: bool | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_refs", tuple(self.source_refs))
        object.__setattr__(self, "metadata", _freeze(self.metadata))
        if (self.kind not in VIEW_KINDS or not isinstance(self.content, str) or not self.content.strip()
                or not self.source_refs or len(set(self.source_refs)) != len(self.source_refs)
                or not self.generator or self.token_cost < 0):
            raise ValueError("invalid node view")

    @property
    def modality(self) -> str:
        return VIEW_KINDS[self.kind]


@dataclass(frozen=True)
class MemoryNode:
    id: str
    retrieval_summary: str
    source_refs: tuple[str, ...]
    canvas: Mapping[str, Any] = field(default_factory=dict)
    source_citations: tuple[tuple[str, str], ...] = ()
    parent: str | None = None
    parent_relation: str | None = None
    related: tuple[str, ...] = ()
    visual_assets: tuple[str, ...] = ()
    version: int = 0
    content_hash: str = ""
    task_signature: str | None = None
    feedback: tuple[FeedbackRecord, ...] = ()
    views: tuple[View, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_refs", tuple(self.source_refs))
        object.__setattr__(self, "canvas", _freeze(self.canvas))
        object.__setattr__(self, "source_citations", tuple(tuple(item) for item in self.source_citations))
        object.__setattr__(self, "related", tuple(self.related))
        object.__setattr__(self, "visual_assets", tuple(self.visual_assets))
        object.__setattr__(self, "feedback", tuple(self.feedback))
        object.__setattr__(self, "views", tuple(self.views))
        if any(not isinstance(view, View) for view in self.views):
            raise ValueError("invalid node view")
        view_keys = [(view.kind, view.content) for view in self.views]
        if (len(set(view_keys)) != len(view_keys)
                or any(not set(view.source_refs) <= set(self.source_refs) for view in self.views)):
            raise ValueError("views must be distinct and cite only the node's sources")
        if (not self.id or not self.retrieval_summary or not self.source_refs
                or len(set(self.source_refs)) != len(self.source_refs) or self.version < 0):
            raise ValueError("invalid memory node")
        if self.task_signature is not None and not self.task_signature.strip():
            raise ValueError("invalid task signature")
        if any(not isinstance(item, FeedbackRecord) for item in self.feedback):
            raise ValueError("invalid feedback record")
        if (self.parent is None and self.parent_relation is not None
                or self.parent_relation is not None
                and (not isinstance(self.parent_relation, str) or not self.parent_relation.strip())
                or self.parent == self.id or self.id in self.related
                or len(set(self.related)) != len(self.related)):
            raise ValueError("invalid memory edge")
        if self.source_citations and (
                len(self.source_citations) != len(self.source_refs)
                or tuple(source for source, _ in self.source_citations) != self.source_refs
                or any(not isinstance(digest, str) or not digest for _, digest in self.source_citations)):
            raise ValueError("invalid source citations")
        expected = self.recompute_hash()
        if self.content_hash and self.content_hash != expected:
            raise ValueError("memory node content hash mismatch")
        object.__setattr__(self, "content_hash", expected)

    def recompute_hash(self) -> str:
        return _digest({"id": self.id, "retrieval_summary": self.retrieval_summary,
                        "source_refs": self.source_refs, "canvas": self.canvas,
                        "source_citations": self.source_citations,
                        "parent": self.parent, "parent_relation": self.parent_relation,
                        "related": self.related,
                        "visual_assets": self.visual_assets, "version": self.version,
                        "task_signature": self.task_signature,
                        "feedback": self.feedback, "views": self.views})

    def view(self, kind: str) -> View | None:
        return next((view for view in self.views if view.kind == kind), None)

    @property
    def image_views(self) -> tuple[View, ...]:
        return tuple(view for view in self.views if view.modality == "image")


@dataclass(frozen=True)
class MemoryPatch:
    id: str
    upsert: tuple[MemoryNode, ...] = ()
    delete: tuple[str, ...] = ()
    expected_revision: int | None = None

    def __post_init__(self) -> None:
        ids = tuple(node.id for node in self.upsert)
        if (not self.id or len(set(ids)) != len(ids) or len(set(self.delete)) != len(self.delete)
                or set(ids) & set(self.delete)):
            raise ValueError("invalid memory patch")


@dataclass(frozen=True)
class PatchReceipt:
    patch_id: str
    revision: int
    snapshot_hash: str


class Memory:
    """Versioned long-term memory with atomic, replayable patches."""

    def __init__(self, sources: Iterable[SourceRecord]):
        source_records = tuple(sources)
        source_map = {source.id: source for source in source_records}
        if not source_map or len(source_map) != len(source_records):
            raise ValueError("sources must have unique stable ids")
        self._sources = source_map
        self._nodes: dict[str, MemoryNode] = {}
        self._patches: list[MemoryPatch] = []
        self._receipts: dict[str, PatchReceipt] = {}
        self._revision = 0
        # ponytail: in-process lock; use durable compare-and-swap once patches cross processes.
        self._lock = RLock()

    @property
    def nodes(self) -> Mapping[str, MemoryNode]:
        return MappingProxyType(self._nodes)

    @property
    def sources(self) -> Mapping[str, SourceRecord]:
        return MappingProxyType(self._sources)

    @property
    def revision(self) -> int:
        return self._revision

    @property
    def snapshot_hash(self) -> str:
        return _digest({"revision": self._revision,
                        "sources": {key: value.content_hash for key, value in sorted(self._sources.items())},
                        "nodes": {key: value.content_hash for key, value in sorted(self._nodes.items())}})

    def apply(self, patch: MemoryPatch) -> PatchReceipt:
        with self._lock:
            return self._apply_unlocked(patch)

    def _apply_unlocked(self, patch: MemoryPatch) -> PatchReceipt:
        if patch.id in self._receipts:
            raise ValueError("patch already applied")
        if patch.expected_revision is not None and patch.expected_revision != self._revision:
            raise ValueError("memory revision conflict")
        candidate = dict(self._nodes)
        for key in patch.delete:
            if key not in candidate:
                raise ValueError("deletion target does not exist")
            del candidate[key]
        for node in patch.upsert:
            candidate[node.id] = node
        self._validate(candidate)
        revision = self._revision + 1
        changed = {node.id for node in patch.upsert}
        committed = {
            key: replace(node, version=revision,
                         source_citations=tuple((ref, self._sources[ref].content_hash) for ref in node.source_refs),
                         content_hash="") if key in changed else node
            for key, node in candidate.items()
        }
        self._nodes = committed
        self._revision = revision
        self._patches.append(patch)
        receipt = PatchReceipt(patch.id, revision, self.snapshot_hash)
        self._receipts[patch.id] = receipt
        return receipt

    def workspace(self, task: TaskSpec, candidates: CandidateSet) -> Workspace:
        with self._lock:
            node_ids = tuple(dict.fromkeys(candidate.node_id for candidate in candidates.candidates
                                           if candidate.node_id is not None))
            if any(node_id not in self._nodes for node_id in node_ids):
                raise ValueError("candidate references an unknown memory node")
            if any(candidate.node_id is not None
                   and candidate.node_version != self._nodes[candidate.node_id].version
                   for candidate in candidates.candidates):
                raise ValueError("candidate node version is stale")
            if any(ref not in self._sources for ref in candidates.source_refs):
                raise ValueError("candidate references an unknown source")
            return Workspace(task, tuple(self._nodes[node_id] for node_id in node_ids),
                             tuple(self._sources[ref] for ref in candidates.source_refs))

    def record_feedback(self, node_id: str, feedback: Iterable[FeedbackRecord],
                        expected_revision: int | None = None, patch_id: str | None = None) -> PatchReceipt:
        """Atomically append optional post-task feedback to one generated node."""
        with self._lock:
            if node_id not in self._nodes:
                raise ValueError("feedback target does not exist")
            additions = tuple(feedback)
            if any(ref not in self._sources for item in additions for ref in item.source_refs):
                raise ValueError("feedback references an unknown source")
            current = self._nodes[node_id]
            return self.apply(MemoryPatch(
                patch_id or f"feedback:{node_id}:{self._revision + 1}",
                expected_revision=self._revision if expected_revision is None else expected_revision,
                upsert=(replace(current, feedback=current.feedback + additions, version=0,
                                 source_citations=(), content_hash=""),),
            ))

    def candidates(self, route: str, identifiers: Iterable[str]) -> tuple[Candidate, ...]:
        """Resolve one independently ranked route against the current memory revision."""
        with self._lock:
            ranked = tuple(identifiers)
            if route not in {"T", "V", "H", "D"} or len(ranked) != len(set(ranked)):
                raise ValueError("invalid retrieval route ranking")
            if route == "D":
                if any(key not in self._sources for key in ranked):
                    raise ValueError("retrieval route references an unknown source")
                return tuple(Candidate(route, rank, (key,)) for rank, key in enumerate(ranked, 1))
            if any(key not in self._nodes for key in ranked):
                raise ValueError("retrieval route references an unknown memory node")
            if route == "V" and any(not self._nodes[key].canvas.get("items")
                                    and not self._nodes[key].visual_assets
                                    and not self._nodes[key].image_views for key in ranked):
                raise ValueError("visual route requires a visual representation")
            return tuple(Candidate(route, rank, self._nodes[key].source_refs, key,
                                   self._nodes[key].version)
                         for rank, key in enumerate(ranked, 1))

    def navigate(self, node_id: str, direction: str) -> tuple[str, ...]:
        """Return only immediate hierarchy neighbors; this is never a global search."""
        with self._lock:
            if node_id not in self._nodes or direction not in {"parent", "children", "related"}:
                raise ValueError("invalid hierarchy navigation")
            node = self._nodes[node_id]
            if direction == "parent":
                return (node.parent,) if node.parent is not None else ()
            if direction == "children":
                return tuple(sorted(key for key, child in self._nodes.items() if child.parent == node_id))
            return node.related

    def retrieve(self, task: TaskSpec, rankings: Mapping[str, Iterable[str]], limit: int | None = None,
                 selected: Iterable[Candidate] | None = None) -> RetrievalResult:
        """Merge independently produced T/V/H/D rankings before optional budget selection."""
        with self._lock:
            if set(rankings) - {"T", "V", "H", "D"}:
                raise ValueError("unknown retrieval route")
            merged = merge_candidates(*(self.candidates(route, rankings.get(route, ()))
                                        for route in ("T", "V", "H", "D")))
            selection = select_candidates(merged, limit, selected)
            workspace = self.workspace(task, CandidateSet(selection.selected))
            return RetrievalResult(merged, selection, workspace)

    def _validate(self, nodes: Mapping[str, MemoryNode]) -> None:
        for node in nodes.values():
            if any(source not in self._sources for source in node.source_refs):
                raise ValueError("unknown source reference")
            if any(ref not in self._sources for item in node.feedback for ref in item.source_refs):
                raise ValueError("feedback references an unknown source")
            expected_citations = tuple((source, self._sources[source].content_hash) for source in node.source_refs)
            if node.source_citations and node.source_citations != expected_citations:
                raise ValueError("source citation version mismatch")
            if node.parent is not None and node.parent not in nodes:
                raise ValueError("unknown parent reference")
            if any(target not in nodes for target in node.related):
                raise ValueError("unknown related reference")
        for start in nodes:
            seen: set[str] = set()
            current: str | None = start
            while current is not None:
                if current in seen:
                    raise ValueError("parent cycle")
                seen.add(current)
                current = nodes[current].parent

    @classmethod
    def replay(cls, sources: Iterable[SourceRecord], patches: Iterable[MemoryPatch]) -> "Memory":
        memory = cls(tuple(sources))
        for patch in patches:
            memory.apply(patch)
        return memory
