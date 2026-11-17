"""Dense text pages: wrapped text at a chosen font, cropped to its content.

The legacy CanvasRenderer always emits a fixed 1024x2048 canvas (2048 vision tokens) however little text
it holds, so smaller fonts only add white space. Here page height follows the text (rounded up to the
reader's 32-pixel patch grid, capped at --max-height), so font size directly sets characters per vision
token. Uses the legacy font and grapheme-preserving pixel wrap; nothing is truncated.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Sequence

GRID = 32


class DensePageRenderer:
    def __init__(self, directory: Path, *, width: int = 1024, font_size: int = 12, max_height: int = 2048,
                 margin: int = 8, line_spacing: float = 1.2) -> None:
        self.directory = Path(directory)
        self.width, self.font_size, self.max_height = width, font_size, max_height
        self.margin, self.step = margin, max(1, math.ceil(font_size * line_spacing))

    def render(self, node_id: str, lines: Sequence[str]) -> list[tuple[str, dict]]:
        from PIL import Image, ImageDraw

        from memorium.runtime.rendering import font, wrap

        face = font(self.font_size)
        wrapped: list[str] = []
        for line in lines:
            if not line.strip():
                continue
            pieces = wrap(line, self.font_size, self.width - 2 * self.margin)
            if pieces is None:
                raise ValueError(f"cannot wrap a line at font {self.font_size}")
            wrapped += [piece for piece in pieces if piece]
        if not wrapped:
            raise ValueError("cannot render an empty view")
        per_page = max(1, (self.max_height - 2 * self.margin) // self.step)
        self.directory.mkdir(parents=True, exist_ok=True)
        pages = []
        for start in range(0, len(wrapped), per_page):
            chunk = wrapped[start:start + per_page]
            height = min(self.max_height, math.ceil((2 * self.margin + len(chunk) * self.step) / GRID) * GRID)
            digest = hashlib.sha256("\n".join([str(self.width), str(self.font_size), *chunk]).encode()).hexdigest()
            path = self.directory / f"{digest[:32]}.png"
            if not path.exists():
                image = Image.new("RGB", (self.width, height), "white")
                draw = ImageDraw.Draw(image)
                for index, text in enumerate(chunk):
                    draw.text((self.margin, self.margin + index * self.step), text, font=face, fill="black",
                              anchor="lt")
                tmp = path.with_suffix(".tmp.png")
                image.save(tmp)
                tmp.replace(path)
            pages.append((str(path), {"pixels": [self.width, height], "lines": len(chunk), "node": node_id,
                                      "page": start // per_page}))
        return pages
