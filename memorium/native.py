"""Native-image (V0) views, attached after a unit is built or loaded from cache.

Deterministic and model-free, and deliberately outside the build-cache key: datasets without images
(LoCoMo) are untouched, so their cached builds stay valid. Oversized photos are downscaled to a fixed
maximum side so one image cannot exhaust the reader's memory budget; copies are content-addressed.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path
from typing import Callable

from .core import MemoryPatch, View
from .unit import BuiltUnit
from .views import node_kind


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bounded_copy(path: str, directory: Path, max_side: int) -> str:
    from PIL import Image, ImageOps

    source = Path(path)
    digest = _sha256(source)
    target = directory / f"{digest[:24]}_{max_side}.png"
    if not target.exists():
        directory.mkdir(parents=True, exist_ok=True)
        with Image.open(source) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            image.thumbnail((max_side, max_side))
            tmp = target.with_suffix(".tmp.png")
            image.save(tmp)
            tmp.replace(target)
    return str(target)


def attach_native_views(unit: BuiltUnit, *, directory: Path, count_image: Callable[[str], int],
                        max_side: int = 640) -> int:
    """Add one V0 view per native image to the segment holding its turn. Returns views added."""
    upserts, added = [], 0
    for node_id in unit.segments:
        node = unit.memory.nodes[node_id]
        views = list(node.views)
        for ref in node.source_refs:
            source = unit.memory.sources[ref]
            ids = list(source.locator.get("image_ids") or [])
            for index, path in enumerate(source.images):
                bounded = bounded_copy(path, directory, max_side)
                if any(v.kind == "V0" and v.content == bounded for v in views):
                    continue
                views.append(View("V0", bounded, (ref,), "native_image", count_image(bounded), True,
                                  {"image_id": ids[index] if index < len(ids) else "", "original": path,
                                   "max_side": max_side}))
                added += 1
        if len(views) != len(node.views):
            upserts.append(replace(node, views=tuple(views), version=0, source_citations=(), content_hash=""))
    if upserts:
        unit.memory.apply(MemoryPatch(f"{unit.unit_id}:native_views:{unit.memory.revision + 1}",
                                      expected_revision=unit.memory.revision, upsert=tuple(upserts)))
    return added


def native_images(node) -> list[View]:
    return [view for view in node.views if view.kind == "V0"] if node_kind(node) != "note" else []
