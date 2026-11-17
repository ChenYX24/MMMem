"""Experience nodes: after every task the system writes one node that summarises how the task was solved.

An experience node is an ordinary memory node (kind "experience") with views and source references:
  T0  text: the task, what was searched (routes, candidates, nodes read), a two-sentence summary written by
      the reader, the outcome, and the evidence the answer rested on, quoted verbatim from the source records;
  V3  optional image: a Qwen-Image picture generated from the summary (text-to-image, never an edit of a
      source photo), marked generated; it travels with the node and the reader may look at it;
  source_refs / related: the exact source records quoted and the nodes that were read, so every claim in
      the node can be followed back to the original record.

Lifecycle (one store per memory unit and experimental arm; never shared):
  staged    a new node first enters the staging area. Staged nodes are searched first on every later task as
            their own retrieval route; their hits always join the candidate pool, and the reranker / RRF decides
            how they rank against ordinary evidence.
  adopted   a later task *adopts* a node when the node was selected for the reader, the reader cited one of its
            source records, and the cited records support the answer. Rephrasings of one question count at most
            `family_cap` times (intent family = question word + named entities, a frozen runtime rule).
  long-term at consolidation (every `period` tasks or when the staging area is full) a staged node adopted
            >= `promote_uses` times across >= `promote_families` families is promoted: it is indexed in the
            unit's main indices and is retrieved like any other node from then on. Staged nodes not adopted
            within `stale_after` tasks expire; overflow evicts the least used, oldest first.
Only public questions, the unit's own sources and the system's earlier outputs are used -- never gold.

Pre-reader admission (Settings.admission, default off = the historical behaviour; needs `outside`):
  An experience node is found by question text alone, but its old answer may belong to another picture. With
  admission on, each node records the sha256 of the picture its own task showed (canvas "query_image_sha256",
  None = no picture), and the top-1 ranked node is admitted *before* it reaches the reader:
    full     the current question shows no picture, or the node's task showed the same picture (sha256):
             T0 and V3 as before;
    pointer  the current question shows a picture and the node's task showed none, another one, or an older
             node recorded none: only Task, Search and the verbatim evidence lines are sent, plus one scope line;
             the old Outcome and Summary lines and the generated picture V3 are left out;
    skip     the node has no source record still in memory (or no verbatim evidence line): nothing is sent.
  A pointer-admitted node never counts as adopted (credit(pointers=...)): its records may help, but its old
  conclusion was not shown, so it gains no promotion credit and no picture. Ranked candidates (staged and
  long-term), the admission decision and the post-answer adoption are logged separately.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import hashlib
from pathlib import Path
import re
from typing import Callable, Sequence

from .core import VIEW_KINDS, MemoryNode, MemoryPatch, View
from .training.rsi_data import supported
from .unit import source_line
from .views import node_kind
from .experience_keys import entities, family

VIEW_KINDS.setdefault("V3", "image")

SUMMARY_PROMPT = (
    "You just answered a question from memory. In at most two sentences, summarise what the memory says that "
    "answers it (or that it says nothing about it), citing record ids in brackets. Do not add facts that are "
    "not in the records.\n\nQuestion: {question}\nYour answer: {answer}\n\nRecords you read:\n{records}\n\nSummary:"
)
POINTER_SCOPE_LINE = ("Scope: that task showed a different picture, so its answer is left out; its records are "
                      "only pointers.")
EVIDENCE_HEAD = "Evidence (verbatim records):"
IMAGE_PROMPT = ("A clear, realistic illustration of this remembered situation, no text in the image: {summary}")
_OPTION = re.compile(r"^\(([A-D])\)\s*(.+)$", re.M)


def image_sha256(path: str | None) -> str | None:
    """Stable identity of a question picture: the sha256 of its bytes (None: the question shows no picture)."""
    if not path:
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def pointer_text(text: str) -> str | None:
    """An experience T0 text as source pointers only: Outcome and Summary lines out, the scope line before the
    evidence. None when no verbatim evidence line follows the evidence header (nothing to point to)."""
    kept = [line for line in text.split("\n") if not line.startswith(("Outcome:", "Summary:"))]
    if EVIDENCE_HEAD not in kept:
        return None
    at = kept.index(EVIDENCE_HEAD)
    if not any(line.strip() for line in kept[at + 1:]):
        return None
    return "\n".join(kept[:at] + [POINTER_SCOPE_LINE] + kept[at:])


def answer_text(question: str, answer: str) -> str:
    """A bare option letter is mapped to its option text so support checks compare words."""
    letter = answer.strip().strip("().").upper()
    if len(letter) == 1:
        for found, text in _OPTION.findall(question):
            if found == letter:
                return text.strip()
    return answer


@dataclass
class Settings:
    top_k: int = 3
    capacity: int = 32
    period: int = 8
    stale_after: int = 32
    promote_uses: int = 3
    promote_families: int = 2
    family_cap: int = 2
    node_tokens: int = 384
    image: str = "adopt"            # off | stage | adopt | promote: when a node gets its generated picture
    image_budget: int = 24          # generated pictures per memory unit
    summary: bool = True            # reader writes the two-sentence summary
    promotion: bool = True          # False: staged only (ablation)
    staging: bool = True            # False: every node is long-term at once (ablation)
    supported_only: bool = False    # True: only nodes whose answer the quoted records support enter staging
    outside: bool = False           # True: top-1 experience node rides outside the evidence token budget
    admission: bool = False         # True: picture-scoped pre-reader admission of the top-1 node (needs outside)


@dataclass
class Record:
    node_id: str
    created: int
    last_adopted: int
    state: str = "staged"           # staged | long_term | expired | evicted
    retrieved: int = 0
    selected: int = 0
    adopted: int = 0
    families: Counter = field(default_factory=Counter)
    admitted: Counter = field(default_factory=Counter)   # admission decisions: full | pointer | skip
    pointer_cited: int = 0          # pointer reads whose citations hit the node's records (never adoption)


class ExperienceStore:
    def __init__(self, unit, staged_index, settings: Settings, *, text_index, visual_index=None,
                 summarize: Callable[[str], str] | None = None, draw: Callable[[str, int], str] | None = None,
                 count_image: Callable[[str], int] | None = None) -> None:
        self.unit, self.index, self.s = unit, staged_index, settings
        self.text_index, self.visual_index = text_index, visual_index
        self.summarize, self.draw, self.count_image = summarize, draw, count_image
        self.records: dict[str, Record] = {}
        self.events: list[dict] = []
        self.images = 0
        self.last_pointer_cited: list[str] = []
        if self.s.admission and not self.s.outside:
            raise ValueError("experience admission applies to the top-1 node outside the budget: needs outside")

    # ---- staged route (u05.run_task(staging=store)) ---------------------------------------------------
    def rank(self, query: str, scope: str, step: int, admitted=lambda node: True) -> list[str]:
        states = ("staged", "long_term") if self.s.outside else ("staged",)
        live = {n for n, r in self.records.items() if r.state in states}
        scores = self.index.scores(query, scope, allowed=lambda n: n in live)
        hits = [n for n, _ in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0])) if admitted(n)][:self.s.top_k]
        for node in hits:
            self.records[node].retrieved += 1
        return hits

    def candidates(self, hits: Sequence[str]) -> list[dict]:
        """The ranked candidates as logged: node, lifecycle state (staged or long_term) and rank."""
        return [{"node": n, "state": self.records[n].state, "rank": i} for i, n in enumerate(hits)]

    def admit(self, node_id: str, query_image_sha256: str | None) -> tuple[dict, list[View]]:
        """Pre-reader admission of one ranked node: (log entry, views to send). See the module docstring."""
        memory = self.unit.memory
        node = memory.nodes[node_id]
        canvas = node.canvas
        recorded = "query_image_sha256" in canvas
        source_sha = canvas.get("query_image_sha256")
        entry = {"node": node_id, "state": self.records[node_id].state if node_id in self.records else None,
                 "query_image_sha256": query_image_sha256,
                 "source_image_sha256": source_sha, "source_image_recorded": recorded}
        refs = tuple(r for r in node.source_refs if r in memory.sources)
        t0 = node.view("T0")
        if not refs or t0 is None:
            decision, reason, views = "skip", "no valid source pointer", []
        elif query_image_sha256 is None:
            decision, reason = "full", "question shows no picture"
        elif recorded and source_sha == query_image_sha256:
            decision, reason = "full", "same picture"
        else:
            decision = "pointer"
            reason = ("source picture not recorded (older node)" if not recorded
                      else "source task showed no picture" if source_sha is None else "different picture")
        if decision == "full":
            views = [v for v in node.views if v.kind in ("T0", "V3")]
        elif decision == "pointer":
            text = pointer_text(t0.content)
            if text is None:
                decision, reason, views = "skip", "no verbatim evidence line", []
            else:
                views = [View("T0", text, node.source_refs, "experience_pointer",
                              self.unit.config.count_tokens(text), None, {"pointer": True})]
        entry.update(decision=decision, reason=reason, views=[v.kind for v in views])
        if node_id in self.records:
            self.records[node_id].admitted[decision] += 1
        return entry, views

    # ---- after a task ---------------------------------------------------------------------------------
    def _lines(self, refs: Sequence[str]) -> list[str]:
        memory, normalize = self.unit.memory, self.unit.config.normalize_time
        return [source_line(memory.sources[r], normalize_time=normalize) for r in refs if r in memory.sources]

    def credit(self, question: str, answer: str, selected: Sequence[str], cited: Sequence[str], step: int,
               pointers: Sequence[str] = ()) -> list[str]:
        """Post-answer adoption. Nodes in `pointers` were shown without their old answer: a citation of their
        records is logged (last_pointer_cited) but never counted as adopting the node's earlier conclusion."""
        said = answer_text(question, answer)
        ok = bool(said.strip()) and bool(cited) and supported(said, self._lines(cited))
        fam, adopted = family(question), []
        self.last_pointer_cited = []
        for node in selected:
            record = self.records.get(node)
            if record is None:
                continue
            record.selected += 1
            hit = bool(set(cited) & set(self.unit.memory.nodes[node].source_refs))
            if node in pointers:
                if hit:
                    record.pointer_cited += 1
                    self.last_pointer_cited.append(node)
                    self.events.append({"step": step, "event": "pointer_cited", "node": node, "supported": ok})
                continue
            if not ok or not hit:
                continue
            if record.families[fam] >= self.s.family_cap:
                continue
            record.families[fam] += 1
            record.adopted += 1
            record.last_adopted = step
            adopted.append(node)
            if self.s.image == "adopt" and record.adopted == 1:
                self._picture(node)
        return adopted

    def record(self, question: str, answer: str, result, step: int, routes: str,
               query_image_sha256: str | None = None) -> str | None:
        """Write the experience node of this task (always), stage it, index it on the staged route.
        With admission on, the node also records the sha256 of this task's own picture (None: no picture)."""
        memory = self.unit.memory
        said = answer_text(question, answer)
        cited = [r for r in result.cited if r in memory.sources]
        is_supported = bool(said.strip()) and bool(cited) and supported(said, self._lines(cited))
        read = [n for n in result.selected if node_kind(memory.nodes[n]) != "experience"]
        refs = cited if cited else [r for n in read[:3] for r in memory.nodes[n].source_refs][:6]
        refs = list(dict.fromkeys(refs))
        if not refs:
            return None  # nothing was read: there is no source an experience could point to
        evidence = self._lines(refs)
        summary = ""
        if self.s.summary and self.summarize is not None and evidence:
            summary = self.summarize(SUMMARY_PROMPT.format(question=question, answer=answer or "insufficient",
                                                           records="\n".join(evidence[:8]))).strip().splitlines()
            summary = summary[0][:400] if summary else ""
        outcome = (f'answered "{answer}"; the quoted records support it' if is_supported
                   else f'answered "{answer}" but the quoted records do not establish it' if answer
                   else "no answer was found in the records read")
        head = [f"Experience from an earlier task (task {step})", f"Task: {question.splitlines()[0]}",
                f"Search: routes {routes}; {len(result.retrieved)} candidates; read {len(read)} nodes",
                f"Outcome: {outcome}"]
        if summary:
            head.append(f"Summary: {summary}")
        head.append("Evidence (verbatim records):")
        lines = list(head)
        count = self.unit.config.count_tokens
        for line in evidence:
            if count("\n".join(lines + [line])) > self.s.node_tokens:
                break
            lines.append(line)
        text = "\n".join(lines)
        node_id = f"{self.unit.unit_id}:experience:{step}"
        canvas = {"type": "experience", "task": question.splitlines()[0], "step": step,
                  "supported": is_supported, "summary": summary,
                  "applicability": sorted(entities(question + " " + " ".join(evidence)))}
        if self.s.admission:
            canvas["query_image_sha256"] = query_image_sha256
        node = MemoryNode(node_id, f"Experience: {question.splitlines()[0]}", tuple(refs),
            canvas=canvas,
            related=tuple(read),
            views=(View("T0", text, tuple(refs), "experience", count(text)),))
        memory.apply(MemoryPatch(f"{node_id}:{memory.revision + 1}", expected_revision=memory.revision, upsert=(node,)))
        self.records[node_id] = Record(node_id, step, step)
        if self.s.supported_only and not is_supported:
            self.records[node_id].state = "kept_unstaged"
        elif self.s.staging:
            self.index.add([(node_id, self.unit.unit_id, text)])
        else:
            self._long_term(node_id, step, "no staging")
        self.events.append({"step": step, "event": "record", "node": node_id, "supported": is_supported})
        if self.s.image == "stage":
            self._picture(node_id)
        staged = sum(r.state == "staged" for r in self.records.values())
        if self.s.staging and staged > self.s.capacity:
            self.consolidate(step, reason="capacity")
        return node_id

    # ---- consolidation --------------------------------------------------------------------------------
    def consolidate(self, step: int, reason: str = "period") -> list[str]:
        promoted = []
        staged = [r for r in self.records.values() if r.state == "staged"]
        if self.s.promotion:
            for r in staged:
                if r.adopted >= self.s.promote_uses and len(r.families) >= self.s.promote_families:
                    self._long_term(r.node_id, step, "adopted")
                    promoted.append(r.node_id)
        for r in staged:
            if r.state == "staged" and step - r.last_adopted > self.s.stale_after:
                r.state = "expired"
                self.events.append({"step": step, "event": "expired", "node": r.node_id})
        excess = sum(r.state == "staged" for r in self.records.values()) - self.s.capacity
        if excess > 0:
            for r in sorted((r for r in self.records.values() if r.state == "staged"),
                            key=lambda r: (r.adopted, r.last_adopted, r.created))[:excess]:
                r.state = "evicted"
                self.events.append({"step": step, "event": "evicted", "node": r.node_id})
        self.events.append({"step": step, "event": "consolidate", "reason": reason, "promoted": promoted})
        return promoted

    def _long_term(self, node_id: str, step: int, why: str) -> None:
        from .u05 import index_node
        self.records[node_id].state = "long_term"
        if self.s.image == "promote":
            self._picture(node_id)
        index_node(self.unit, node_id, self.text_index, self.visual_index)
        self.events.append({"step": step, "event": "long_term", "node": node_id, "why": why})

    def _picture(self, node_id: str) -> None:
        if self.draw is None or self.images >= self.s.image_budget:
            return
        node = self.unit.memory.nodes[node_id]
        if node.view("V3") is not None:
            return
        scene = str(node.canvas.get("summary") or node.canvas.get("task") or "")
        if not scene:
            return
        path = self.draw(IMAGE_PROMPT.format(scene=scene, summary=scene), int(node_id.rsplit(":", 1)[-1]))
        self.images += 1
        cost = self.count_image(path) if self.count_image else 256
        view = View("V3", path, node.source_refs, "qwen_image_experience", cost, False,
                    {"generated": True, "prompt": scene})
        from .unit import attach_view
        attach_view(self.unit.memory, node_id, view, f"{node_id}:picture:{self.unit.memory.revision + 1}")
        self.events.append({"node": node_id, "event": "picture", "path": path})

    def summary(self) -> dict:
        states = Counter(r.state for r in self.records.values())
        out = {"nodes": len(self.records), "states": dict(states), "images": self.images,
               "adopted": sum(r.adopted for r in self.records.values()),
               "retrieved": sum(r.retrieved for r in self.records.values())}
        if self.s.admission:
            out["admitted"] = dict(sum((r.admitted for r in self.records.values()), Counter()))
            out["pointer_cited"] = sum(r.pointer_cited for r in self.records.values())
        return out


# Warm-up questions have no official prompt template (LooGLE keeps one per task type in each official question):
# a generic one with the same placeholders is used instead. The question text already holds its four options.
WARMUP_INSTRUCTION = "Answer the question using only the context.\n\nContext:\n{context}\n\n{options}Question: {question}"


def warmup_tasks(rows, unit_id: str) -> list[dict]:
    """Self-generated questions of one unit as runner tasks, played before the scored questions.

    Rows come from scripts/run_memorium_joint_rsi.py --warmup: they were written from the unit's own source
    records only. Their answers are not kept, so nothing here can be scored or leak into the answer path."""
    tasks = []
    for row in rows:
        if row["unit"] != unit_id:
            continue
        if not row.get("warmup"):
            raise ValueError("not a warm-up row: refusing to play it as a task")
        tasks.append({"id": f"warmup:{unit_id}:{len(tasks)}", "sample_id": unit_id, "question_index": -1,
                      "question": row["question"], "options": row.get("options") or [], "warmup": True,
                      "instruction": WARMUP_INSTRUCTION, "loogle_question": row["question"],
                      "loogle_options": []})
    return tasks
