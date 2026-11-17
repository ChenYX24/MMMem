"""Build once, reuse everywhere: a content-addressed cache of built memory units.

A unit's build (structure + T1/V1/V2 views) depends only on its sources, the view configuration, the
frozen writer/image models, the font and the build code, not on the arm's selector, reader or write-back.
So arms that share those inputs load the same replayable patch log instead of regenerating it. Loading
replays the patches and must reproduce the saved snapshot hash, and every image a view points to must
still have its recorded sha256.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Callable, Iterable, Mapping

from .core import FeedbackRecord, Memory, MemoryNode, MemoryPatch, SourceRecord, View, _canonical
from .unit import BuiltUnit, UnitConfig
from .views import ViewLedger

ROOT = Path(__file__).resolve().parents[1]
BUILD_CODE = ("memorium/core.py", "memorium/unit.py", "memorium/structure.py", "memorium/views.py",
              "memorium/materialize.py", "memorium/adapters/render.py", "memorium/adapters/imagegen.py",
              "memorium/cache.py")


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def node_to_json(node: MemoryNode) -> dict:
    return _canonical(node)


def node_from_json(data: Mapping[str, Any]) -> MemoryNode:
    fields = dict(data)
    fields["feedback"] = tuple(FeedbackRecord(**item) for item in fields.get("feedback", ()))
    fields["views"] = tuple(View(**item) for item in fields.get("views", ()))
    fields["source_citations"] = tuple(tuple(item) for item in fields.get("source_citations", ()))
    return MemoryNode(**fields)


def patch_to_json(patch: MemoryPatch) -> dict:
    return {"id": patch.id, "upsert": [node_to_json(node) for node in patch.upsert],
            "delete": list(patch.delete), "expected_revision": patch.expected_revision}


def patch_from_json(data: Mapping[str, Any]) -> MemoryPatch:
    return MemoryPatch(data["id"], tuple(node_from_json(node) for node in data["upsert"]),
                       tuple(data["delete"]), data["expected_revision"])


def build_key(*, sources_sha: str, unit_config: UnitConfig, structure: bool, writer: str | None,
              image_model: str | None, font: str | None, dry_run: bool) -> tuple[str, dict]:
    """Everything the build depends on, and nothing it does not (selector, reader, write-back)."""
    views = asdict(unit_config.views)
    spec = {
        "sources_sha256": sources_sha, "views": views, "structure": structure,
        "normalize_time": unit_config.normalize_time, "segment_tokens": unit_config.segment_tokens,
        "writer": writer, "image_model": image_model if views.get("v2") else None,
        "font_sha256": sha256_file(font) if font and os.path.isfile(font) else font,
        "code_sha256": {path: sha256_file(ROOT / path) for path in BUILD_CODE},
        "dry_run": dry_run,
    }
    key = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:24]
    return key, spec


class BuildCache:
    def __init__(self, root: Path, key: str, spec: dict) -> None:
        self.dir = Path(root) / key
        self.dir.mkdir(parents=True, exist_ok=True)
        spec_path = self.dir / "spec.json"
        if not spec_path.exists():
            spec_path.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        elif json.loads(spec_path.read_text(encoding="utf-8")) != spec:
            raise ValueError(f"build cache {self.dir} was written for a different spec")
        self.key = key
        self.views_dir = self.dir / "views"

    def _path(self, unit_id: str) -> Path:
        return self.dir / "units" / f"{unit_id}.json"

    def save(self, unit: BuiltUnit) -> dict:
        images = {view.content: sha256_file(view.content)
                  for node in unit.memory.nodes.values() for view in node.image_views if os.path.isfile(view.content)}
        payload = {
            "schema": "memorium-u05-build-cache-v1", "unit": unit.unit_id,
            "patches": [patch_to_json(patch) for patch in unit.memory._patches],
            "snapshot_hash": unit.memory.snapshot_hash, "revision": unit.memory.revision,
            "segments": unit.segments, "sessions": unit.sessions, "lines": unit.lines,
            "ledger": {"v2_generated": unit.ledger.v2_generated, "uses": unit.ledger.uses},
            "images_sha256": images, "saved_at": time.time(),
        }
        path = self._path(unit.unit_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, path)  # atomic: a reader never sees a half-written unit
        return {"snapshot_hash": payload["snapshot_hash"], "images": len(images)}

    def load(self, unit_id: str, sources: Iterable[SourceRecord], config: UnitConfig) -> BuiltUnit | None:
        path = self._path(unit_id)
        if not path.is_file():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        memory = Memory.replay(sources, (patch_from_json(patch) for patch in payload["patches"]))
        if memory.snapshot_hash != payload["snapshot_hash"]:
            raise ValueError(f"cached build of {unit_id} does not replay to its recorded snapshot")
        for image, digest in payload["images_sha256"].items():
            if not os.path.isfile(image) or sha256_file(image) != digest:
                raise ValueError(f"cached build of {unit_id} references a missing or changed image: {image}")
        ledger = ViewLedger(uses=dict(payload["ledger"]["uses"]), v2_generated=payload["ledger"]["v2_generated"])
        return BuiltUnit(unit_id, memory, list(payload["segments"]), list(payload["sessions"]),
                         {key: list(value) for key, value in payload["lines"].items()}, ledger, config)

    def get_or_build(self, unit_id: str, sources: list[SourceRecord], config: UnitConfig,
                     build: Callable[[], BuiltUnit], wait_seconds: int = 6 * 3600) -> tuple[BuiltUnit, str]:
        """Load if present; else build under an exclusive lock so two arms never build the same unit twice."""
        cached = self.load(unit_id, sources, config)
        if cached is not None:
            return cached, "hit"
        lock = self._path(unit_id).with_suffix(".lock")
        lock.parent.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            deadline = time.time() + wait_seconds
            while lock.exists() and time.time() < deadline:
                time.sleep(30)
            cached = self.load(unit_id, sources, config)
            if cached is None:
                raise TimeoutError(f"another process holds the build lock for {unit_id} and produced nothing")
            return cached, "hit_after_wait"
        try:
            os.write(fd, f"{os.getpid()}@{os.uname().nodename}".encode())
            unit = build()
            self.save(unit)
            return unit, "built"
        finally:
            os.close(fd)
            lock.unlink(missing_ok=True)
