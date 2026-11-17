"""One memory unit (a conversation or scenario) built into a multi-view node graph.

Node kinds: segment (retrieval leaf), session (time chain), entity, timeline, note (task write-back).
Views are optional per node and decided by memorium.views.MaterializationPolicy.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field, replace
import hashlib
import re
from typing import Callable, Iterable, Mapping, Sequence

from .core import Memory, MemoryNode, MemoryPatch, SourceRecord, View
from .structure import (Turn, annotate_relative_time, candidate_entities, estimate_tokens, parse_anchor,
                        segment_session, session_date)
from .views import MaterializationPolicy, ViewConfig, ViewLedger, node_kind

_RELATIVE = re.compile(r"\[= ")


def _short(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()[:12]


def source_line(source: SourceRecord, *, normalize_time: bool) -> str:
    locator = source.locator
    text = source.text
    if normalize_time:
        text = annotate_relative_time(text, parse_anchor(locator.get("date_time")))
    caption = locator.get("visual_caption")
    if caption:
        text = f"{text} [shared image: {caption}]"
    who = locator.get("speaker") or "?"
    when = session_date(locator.get("date_time"))
    return f"[{locator.get('dia_id') or source.id}] {who} ({when}): {text}"


@dataclass
class UnitConfig:
    views: ViewConfig = field(default_factory=ViewConfig)
    normalize_time: bool = True
    structure: bool = True
    segment_tokens: int = 400
    count_tokens: Callable[[str], int] = estimate_tokens


@dataclass
class BuiltUnit:
    unit_id: str
    memory: Memory
    segments: list[str]
    sessions: list[str]
    lines: dict[str, list[str]]
    ledger: ViewLedger
    config: UnitConfig


def _session_key(source: SourceRecord) -> tuple[int, str]:
    locator = source.locator
    index = locator.get("session_index", locator.get("session_id", 0))
    return (int(index) if str(index).isdigit() else 0, str(index))


def build_unit(unit_id: str, sources: Sequence[SourceRecord], config: UnitConfig) -> BuiltUnit:
    """Deterministic, model-free construction of the unit's structure plus T0 views.

    T1/V1/V2 views are attached afterwards by materializers, so this stays CPU-only and testable.
    """
    if not sources:
        raise ValueError("empty memory unit")
    memory = Memory(sources)
    by_session: dict[tuple[int, str], list[SourceRecord]] = defaultdict(list)
    for source in sorted(sources, key=lambda item: item.ordinal):
        by_session[_session_key(source)].append(source)

    nodes: list[MemoryNode] = []
    lines: dict[str, list[str]] = {}
    segment_ids: list[str] = []
    session_ids: list[str] = []
    previous_session: str | None = None
    for key in sorted(by_session):
        members = by_session[key]
        session_id = f"{unit_id}:session:{key[1]}"
        date_text = session_date(members[0].locator.get("date_time"))
        turns = [Turn(source.id, source.ordinal, session_id,
                      source_line(source, normalize_time=config.normalize_time)) for source in members]
        child_ids: list[str] = []
        for index, segment in enumerate(segment_session(turns, max_tokens=config.segment_tokens,
                                                        count_tokens=config.count_tokens)):
            segment_id = f"{session_id}:seg:{index}"
            text_lines = [turn.text for turn in segment]
            refs = tuple(turn.id for turn in segment)
            nodes.append(MemoryNode(
                segment_id, "\n".join(text_lines), refs, canvas={"type": "segment", "session": session_id},
                parent=session_id, parent_relation="segment_of",
                views=(View("T0", "\n".join(text_lines), refs, "source", config.count_tokens("\n".join(text_lines))),),
            ))
            lines[segment_id] = text_lines
            child_ids.append(segment_id)
            segment_ids.append(segment_id)
        refs = tuple(source.id for source in members)
        session_lines = [f"Session {key[1]} on {date_text}"] + [turn.text for turn in turns]
        nodes.append(MemoryNode(
            session_id, f"Session {key[1]} on {date_text}", refs,
            canvas={"type": "session", "date": date_text, "children": child_ids},
            parent=previous_session, parent_relation="follows" if previous_session else None,
        ))
        lines[session_id] = session_lines
        session_ids.append(session_id)
        previous_session = session_id

    # Parents must exist before children reference them, so sessions are applied first.
    sessions = [node for node in nodes if node_kind(node) == "session"]
    segments = [node for node in nodes if node_kind(node) == "segment"]
    memory.apply(MemoryPatch(f"{unit_id}:build:sessions", upsert=tuple(sessions)))
    memory.apply(MemoryPatch(f"{unit_id}:build:segments", upsert=tuple(segments)))

    if config.structure:
        _add_structure(unit_id, memory, sources, lines, segment_ids, config)
    return BuiltUnit(unit_id, memory, segment_ids, session_ids, lines, ViewLedger(), config)


def _add_structure(unit_id: str, memory: Memory, sources: Sequence[SourceRecord], lines: dict[str, list[str]],
                   segment_ids: list[str], config: UnitConfig) -> None:
    """Entity nodes (who/what recurs) and one timeline node (dated events), both linked to segments."""
    segment_of = {ref: segment for segment in segment_ids for ref in memory.nodes[segment].source_refs}
    speakers = {str(source.locator.get("speaker")) for source in sources if source.locator.get("speaker")}
    turns = [Turn(source.id, source.ordinal, "", source.text) for source in sources]
    extra: list[MemoryNode] = []
    for name, refs in sorted(candidate_entities(turns, speakers).items()):
        linked = tuple(dict.fromkeys(segment_of[ref] for ref in refs if ref in segment_of))
        if not linked:
            continue
        node_id = f"{unit_id}:entity:{_short(name)}"
        entity_lines = [f"Entity: {name}"] + [source_line(memory.sources[ref], normalize_time=config.normalize_time)
                                               for ref in refs]
        extra.append(MemoryNode(node_id, f"Entity: {name}", tuple(refs), canvas={"type": "entity", "name": name},
                                related=linked,
                                views=(View("T0", "\n".join(entity_lines), tuple(refs), "structure",
                                            config.count_tokens("\n".join(entity_lines))),)))
        lines[node_id] = entity_lines

    dated = [source for source in sorted(sources, key=lambda item: item.ordinal)
             if _RELATIVE.search(annotate_relative_time(source.text, parse_anchor(source.locator.get("date_time"))))]
    if dated:
        node_id = f"{unit_id}:timeline"
        timeline_lines = ["Timeline of dated events"] + [source_line(source, normalize_time=True) for source in dated]
        refs = tuple(source.id for source in dated)
        linked = tuple(dict.fromkeys(segment_of[ref] for ref in refs if ref in segment_of))
        extra.append(MemoryNode(node_id, "Timeline of dated events", refs, canvas={"type": "timeline"},
                                related=linked,
                                views=(View("T0", "\n".join(timeline_lines), refs, "structure",
                                            config.count_tokens("\n".join(timeline_lines))),)))
        lines[node_id] = timeline_lines
    if extra:
        memory.apply(MemoryPatch(f"{unit_id}:build:structure", upsert=tuple(extra)))


def attach_view(memory: Memory, node_id: str, view: View, patch_id: str) -> None:
    """Add or replace one view as a versioned, replayable patch."""
    node = memory.nodes[node_id]
    kept = tuple(item for item in node.views if item.kind != view.kind)
    memory.apply(MemoryPatch(patch_id, expected_revision=memory.revision,
                             upsert=(replace(node, views=kept + (view,), version=0, source_citations=(),
                                             content_hash=""),)))


def text_of(node: MemoryNode) -> str:
    view = node.view("T1") or node.view("T0")
    return view.content if view else node.retrieval_summary


def retrieval_texts(node: MemoryNode) -> list[str]:
    """Text items the T route embeds: every text view, each already under the segment token cap."""
    return [view.content for view in node.views if view.modality == "text"] or [node.retrieval_summary]


def image_items(node: MemoryNode) -> list[str]:
    return [view.content for view in node.image_views]


NOTE_PREFIX = "note"


def note_id(unit_id: str, question: str) -> str:
    return f"{unit_id}:{NOTE_PREFIX}:{_short(question.strip().lower())}"


def write_note(memory: Memory, unit_id: str, question: str, answer: str, cited: Sequence[str],
               used_nodes: Sequence[str], *, count_tokens: Callable[[str], int] = estimate_tokens,
               style: str = "answer", normalize_time: bool = True) -> tuple[str, bool]:
    """Post-reader write-back. Returns (node_id, updated). A repeated question versions the same node."""
    refs = tuple(dict.fromkeys(ref for ref in cited if ref in memory.sources))
    if not refs:
        refs = tuple(dict.fromkeys(ref for node in used_nodes for ref in memory.nodes[node].source_refs))[:6]
    if not refs or not answer.strip():
        raise ValueError("a note needs an answer and at least one cited source")
    node_id = note_id(unit_id, question)
    text = f"Earlier question: {question}\nModel's earlier answer (unverified): {answer.strip()}"
    if style == "evidence":
        # The note becomes a compact pointer: the exact records the answer rests on travel with it.
        text += "\nRecords it cited:\n" + "\n".join(
            source_line(memory.sources[ref], normalize_time=normalize_time) for ref in refs[:4])
    # Link to the segments that hold the cited evidence first, so a reader can verify the note.
    holding = [node for node, item in memory.nodes.items()
               if str(item.canvas.get("type")) == "segment" and set(item.source_refs) & set(refs)]
    related = tuple(dict.fromkeys(node for node in holding + list(used_nodes)
                                  if node in memory.nodes and node != node_id))
    updated = node_id in memory.nodes
    node = MemoryNode(node_id, text, refs, canvas={"type": "note", "question": question}, related=related,
                      views=(View("T0", text, refs, "reader_note", count_tokens(text)),))
    memory.apply(MemoryPatch(f"{node_id}:{memory.revision + 1}", expected_revision=memory.revision,
                             upsert=(node,)))
    return node_id, updated


def link_note_to_entities(memory: Memory, note: str) -> list[str]:
    """Merge a note into entity nodes that share its evidence: add it to their related links (versioned)."""
    note_refs = set(memory.nodes[note].source_refs)
    touched = []
    for node_id, node in list(memory.nodes.items()):
        if node_kind(node) != "entity" or note in node.related or not note_refs & set(node.source_refs):
            continue
        memory.apply(MemoryPatch(f"{node_id}:link:{note}:{memory.revision + 1}", expected_revision=memory.revision,
                                 upsert=(replace(node, related=node.related + (note,), version=0,
                                                 source_citations=(), content_hash=""),)))
        touched.append(node_id)
    return touched
