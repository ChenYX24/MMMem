"""Model-backed view materialization: T1 compression, V1 rendered pages, V2 integrated images.

Every generated view must be grounded: T1 keeps only facts citing the node's own sources, V2 must
pass a vision check against its spec. Failures are recorded, never silently accepted.
"""
from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any

from .core import View
from .unit import BuiltUnit, attach_view
from .views import MaterializationPolicy, node_kind

T1_PROMPT = (
    "Rewrite these conversation records as a compact list of facts. One fact per line, each ending with "
    "the record id in brackets, e.g. 'Caroline attended an LGBTQ support group on 7 May 2023 [D1:3]'. "
    "Use absolute dates where the records give them. Keep names, numbers and dates exact. Do not add "
    "anything that is not stated. Output only the list.\n\nRecords:\n"
)
_FACT = re.compile(r"^\s*[-*\d.)]*\s*(.+?)\s*\[([^\]]+)\]\s*$")


class Materializer:
    def __init__(self, directory: Path, *, reader_calls: Any, image_model: str | None,
                 check_model: Any | None = None) -> None:
        self.directory = Path(directory)
        self.calls = reader_calls
        self.check = check_model or reader_calls
        self.image_model = image_model
        self._renderer = None
        self._imagegen = None
        self.log = self.directory / "materialize.jsonl"
        self.directory.mkdir(parents=True, exist_ok=True)

    @property
    def renderer(self):
        if self._renderer is None:
            from .adapters.render import PageRenderer
            self._renderer = PageRenderer(self.directory / "pages")
        return self._renderer

    @property
    def imagegen(self):
        if self._imagegen is None:
            from .adapters.imagegen import QwenImageGenerator
            self._imagegen = QwenImageGenerator(self.image_model, self.directory / "v2")
        return self._imagegen

    def _record(self, row: dict) -> None:
        with self.log.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    def _count(self, text: str, images: tuple = ()) -> int:
        return self.calls.count(text, images) if self.calls is not None else len(text.split())

    def t1(self, unit: BuiltUnit, node_id: str) -> View | None:
        node = unit.memory.nodes[node_id]
        lines = unit.lines.get(node_id) or [node.retrieval_summary]
        row = self.calls.call("writer", T1_PROMPT + "\n".join(lines), max_new_tokens=512)
        ids = {str(unit.memory.sources[ref].locator.get("dia_id") or ref): ref for ref in node.source_refs}
        kept, dropped = [], 0
        for line in str(row.get("output", "")).splitlines():
            match = _FACT.match(line)
            if match and match.group(2).strip() in ids:
                kept.append((match.group(1), ids[match.group(2).strip()]))
            elif line.strip():
                dropped += 1
        self._record({"stage": "T1", "node": node_id, "kept": len(kept), "dropped": dropped,
                      "call": row.get("call")})
        if not kept:
            return None
        text = "\n".join(f"{fact} [{unit.memory.sources[ref].locator.get('dia_id') or ref}]" for fact, ref in kept)
        refs = tuple(dict.fromkeys(ref for _, ref in kept))
        return View("T1", text, refs, "reader_t1", self._count(text), verified=True,
                    metadata={"dropped_lines": dropped})

    def v1(self, unit: BuiltUnit, node_id: str) -> View | None:
        node = unit.memory.nodes[node_id]
        lines = unit.lines.get(node_id)
        if not lines:
            return None
        pages = self.renderer.render(node_id, lines)
        path = pages[0][0]
        tokens = self._count("", (path,))
        self._record({"stage": "V1", "node": node_id, "pages": len(pages), "vision_tokens": tokens})
        return View("V1", path, node.source_refs, "canvas_renderer", tokens, verified=True,
                    metadata={"pages": [p for p, _ in pages]})

    def v2(self, unit: BuiltUnit, node_id: str) -> View | None:
        from .adapters.imagegen import integrate

        node = unit.memory.nodes[node_id]
        lines = unit.lines.get(node_id) or [node.retrieval_summary]
        dia = {str(unit.memory.sources[ref].locator.get("dia_id") or ref): ref for ref in node.source_refs}
        result = integrate(
            lines, list(dia),
            write=lambda prompt: str(self.calls.call("writer", prompt, max_new_tokens=512).get("output", "")),
            draw=lambda scene, seed: self.imagegen(scene, seed),
            look=lambda path, prompt: str(self.check.call("verifier", prompt, images=(path,),
                                                           max_new_tokens=256).get("output", "")),
            seed=int(__import__("hashlib").sha256(node_id.encode()).hexdigest()[:6], 16))
        unit.ledger.v2_generated += 1
        self._record({"stage": "V2", "node": node_id, "verified": result.verified, "attempts": result.attempts})
        if not result.verified or result.path is None:
            return None
        cited = tuple(dict.fromkeys(dia[source] for _, source in result.spec.facts if source in dia))
        return View("V2", result.path, cited or node.source_refs, "qwen_image", self._count("", (result.path,)),
                    verified=True, metadata={"scene": result.spec.scene, "spec_sha256": result.spec.digest()})

    def build_views(self, unit: BuiltUnit, policy: MaterializationPolicy) -> None:
        for node_id in list(unit.memory.nodes):
            node = unit.memory.nodes[node_id]
            if node_kind(node) in {"session", "note"}:
                continue
            text = (node.view("T0").content if node.view("T0") else node.retrieval_summary)
            tokens = self._count(text)
            wanted = []
            if policy.wants_t1(node, tokens, "[= " in text):
                wanted.append("T1")
            if policy.wants_v1(node, tokens):
                wanted.append("V1")
            if policy.wants_v2_at_build(node, unit.ledger):
                wanted.append("V2")
            for kind in wanted:
                view = self.upgrade(unit, node_id, kind)
                if view is not None:
                    attach_view(unit.memory, node_id, view, f"{node_id}:build:{kind}")

    def upgrade(self, unit: BuiltUnit, node_id: str, kind: str) -> View | None:
        return {"T1": self.t1, "V1": self.v1, "V2": self.v2}[kind](unit, node_id)
