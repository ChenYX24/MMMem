from pathlib import Path
import tempfile
import unittest

from memorium.cache import BuildCache, build_key, node_from_json, node_to_json
from memorium.core import FeedbackRecord, MemoryNode, View
from memorium.unit import UnitConfig, attach_view, build_unit
from memorium.views import ViewConfig

from test_memorium_u05 import unit_sources


def built(tmp: Path):
    unit = build_unit("conv-x", unit_sources(), UnitConfig())
    image = tmp / "page.png"
    image.write_bytes(b"png-bytes")
    segment = unit.segments[0]
    attach_view(unit.memory, segment, View("V1", str(image), unit.memory.nodes[segment].source_refs, "canvas", 300),
                "build:v1")
    attach_view(unit.memory, segment, View("T1", "Susie and Seraphim are snakes [D1:2]",
                                           ("locomo:conv-x:D1:2",), "writer", 9), "build:t1")
    unit.ledger.v2_generated = 3
    return unit, image


def key(**overrides):
    args = dict(sources_sha="abc", unit_config=UnitConfig(), structure=True, writer="m|qwen35",
                image_model="img", font=None, dry_run=False)
    args.update(overrides)
    return build_key(**args)


class CacheTest(unittest.TestCase):
    def test_node_json_round_trip_keeps_the_content_hash(self):
        node = MemoryNode("n", "summary", ("s1",), canvas={"type": "note", "items": [{"a": 1}]},
                          related=("m",), views=(View("T0", "text", ("s1",), "source", 3, True, {"k": [1]}),),
                          feedback=(FeedbackRecord("note", "fine", ("s1",)),))
        self.assertEqual(node_from_json(node_to_json(node)).content_hash, node.content_hash)

    def test_saved_unit_replays_to_the_same_memory_and_ledger(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            unit, _ = built(tmp)
            cache = BuildCache(tmp / "cache", *key())
            cache.save(unit)
            loaded = cache.load("conv-x", unit_sources(), UnitConfig())
            self.assertEqual(loaded.memory.snapshot_hash, unit.memory.snapshot_hash)
            self.assertEqual(loaded.memory.revision, unit.memory.revision)
            self.assertEqual((loaded.segments, loaded.lines, loaded.ledger.v2_generated),
                             (unit.segments, unit.lines, 3))

    def test_get_or_build_builds_once_then_hits(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            cache = BuildCache(tmp / "cache", *key())
            calls = []

            def build():
                calls.append(1)
                return built(tmp)[0]

            _, first = cache.get_or_build("conv-x", unit_sources(), UnitConfig(), build)
            _, second = cache.get_or_build("conv-x", unit_sources(), UnitConfig(), build)
            self.assertEqual((first, second, len(calls)), ("built", "hit", 1))

    def test_a_changed_image_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            unit, image = built(tmp)
            cache = BuildCache(tmp / "cache", *key())
            cache.save(unit)
            image.write_bytes(b"different")
            with self.assertRaisesRegex(ValueError, "missing or changed image"):
                cache.load("conv-x", unit_sources(), UnitConfig())

    def test_different_sources_are_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            unit, _ = built(tmp)
            cache = BuildCache(tmp / "cache", *key())
            cache.save(unit)
            with self.assertRaises(ValueError):
                cache.load("conv-x", unit_sources()[:-1], UnitConfig())

    def test_key_tracks_build_inputs_but_not_arm_choices(self):
        base = key()[0]
        self.assertEqual(base, key()[0])
        self.assertNotEqual(base, key(unit_config=UnitConfig(views=ViewConfig(t1=True)))[0])
        self.assertNotEqual(base, key(structure=False)[0])
        self.assertNotEqual(base, key(sources_sha="other")[0])
        self.assertNotEqual(base, key(writer="other|qwen35")[0])
        # The image model only matters when V2 is on.
        self.assertEqual(base, key(image_model="another")[0])

    def test_a_directory_written_for_another_spec_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            k, spec = key()
            BuildCache(Path(tmp), k, spec)
            with self.assertRaisesRegex(ValueError, "different spec"):
                BuildCache(Path(tmp), k, {**spec, "structure": False})


if __name__ == "__main__":
    unittest.main()
