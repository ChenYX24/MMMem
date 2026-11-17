"""Render source text as lossless page images."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence


class PageRenderer:
    """Render ordered text lines into one or more PNG pages without truncating any text."""

    def __init__(self, directory: Path, *, width: int = 1024, font_size: int = 20, lines_per_page: int = 48) -> None:
        from memorium.runtime.inference import CanvasRenderer

        self.renderer = CanvasRenderer(directory)
        self.width, self.font_size, self.lines_per_page = width, font_size, lines_per_page

    def render(self, node_id: str, lines: Sequence[str]) -> list[tuple[str, dict]]:
        # The renderer rejects whitespace-only items; blank lines carry no content to lose.
        lines = [line for line in lines if line.strip()]
        if not lines:
            raise ValueError("cannot render an empty view")
        pages = []
        for start in range(0, len(lines), self.lines_per_page):
            chunk = lines[start:start + self.lines_per_page]
            rows = [[{"id": f"l{index}", "kind": "text", "text": line, "font_size": self.font_size}]
                    for index, line in enumerate(chunk)]
            canvas = self.renderer.compile_flow({"layout": "rows", "width": self.width, "rows": rows})
            path, audit = self.renderer.render({"id": f"{node_id}#p{start // self.lines_per_page}",
                                                "canvas": canvas})
            pages.append((str(path), {"pixels": audit.get("pixels"), "image_sha256": audit.get("image_sha256")}))
        return pages
