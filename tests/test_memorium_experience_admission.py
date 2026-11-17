"""Pre-reader experience admission (memorium/experience.py, Settings.admission; runner --exp-admission).

Built unit + hashing encoder + scripted reader: needs numpy, so it runs in the cluster environment like
tests/test_memorium_experience.py. The pure-text checks at the bottom run anywhere.
"""
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
Q = "What are Melanie's snakes called?"


class AdmissionTest(unittest.TestCase):
    def setUp(self):
        try:
            from tests.test_memorium_u05 import unit_sources
            from memorium.adapters.route_index import HashEncoder, RouteIndex
        except ImportError as exc:
            self.skipTest(str(exc))
        from memorium.u05 import build_indices
        from memorium.unit import UnitConfig, build_unit
        self.unit = build_unit("conv-x", unit_sources(), UnitConfig())
        self.RouteIndex, self.HashEncoder = RouteIndex, HashEncoder
        self.index = RouteIndex(HashEncoder())
        build_indices(self.unit, self.index, None)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.pic = {}
        for name in ("red", "blue"):
            path = Path(self.tmp.name) / f"{name}.png"
            path.write_bytes(name.encode() * 64)  # distinct bytes, distinct sha256
            self.pic[name] = str(path)
        self.seen = []

    def store(self, **kw):
        from memorium.experience import ExperienceStore, Settings
        kw.setdefault("image", "stage")  # every node gets its generated picture V3 at once
        kw.setdefault("outside", True)
        self.drawn = []
        return ExperienceStore(self.unit, self.RouteIndex(self.HashEncoder()), Settings(**kw), text_index=self.index,
                               summarize=lambda p: "Melanie's snakes are Susie and Seraphim [D1:2].",
                               draw=lambda scene, seed: self.drawn.append(scene) or f"/tmp/exp{seed}.png",
                               count_image=lambda path: 100)

    def reader(self, prompt, views):
        """Copies an earlier answer when one is shown (the failure mode); otherwise reads the records."""
        self.seen.append(list(views))
        text = "\n".join(v.content for v in views if v.modality == "text")
        if 'Outcome: answered "' in text:
            old = text.split('Outcome: answered "', 1)[1].split('"', 1)[0]
            return {"output": f"Answer: {old}\nEvidence: D1:2", "status": "OK"}
        if "Susie" in text:
            return {"output": "Answer: Susie and Seraphim\nEvidence: D1:2", "status": "OK"}
        return {"output": "Answer: insufficient\nNeed: more", "status": "OK"}

    def ask(self, store, step, picture=None, question=Q, record=True):
        from memorium import TaskSpec
        from memorium.experience import image_sha256
        from memorium.u05 import ArmConfig, run_task
        arm = ArmConfig(routes=("T",), write_back=False, token_budget=3000)
        path = self.pic.get(picture)
        r = run_task(self.unit, TaskSpec(question), f"q{step}", arm, text_index=self.index, visual_index=None,
                     reader=self.reader, staging=store, step=step, query_image=path)
        decision = (r.experience_admission or {}).get("admission") or {}
        pointers = {decision["node"]} if decision.get("decision") == "pointer" else set()
        adopted = store.credit(question, r.answer, r.selected, r.cited, step, pointers=pointers)
        node = store.record(question, r.answer, r, step, "T", query_image_sha256=image_sha256(path)) if record else None
        return r, node, adopted

    def sent(self):
        return self.seen[-1]

    # ---- decisions ------------------------------------------------------------------------------------
    def test_same_question_other_picture_gets_pointers_only(self):
        s = self.store(admission=True)
        _, first, _ = self.ask(s, 0, picture="red")
        from memorium.experience import image_sha256
        self.assertEqual(self.unit.memory.nodes[first].canvas["query_image_sha256"], image_sha256(self.pic["red"]))
        self.assertIsNotNone(self.unit.memory.nodes[first].view("V3"))
        r, _, adopted = self.ask(s, 1, picture="blue")
        log = r.experience_admission
        self.assertEqual(log["admission"]["node"], first)
        self.assertEqual(log["admission"]["decision"], "pointer")
        self.assertEqual(log["admission"]["reason"], "different picture")
        texts = [v.content for v in self.sent() if v.modality == "text"]
        exp_text = [t for t in texts if t.startswith("Experience from an earlier task")]
        self.assertEqual(len(exp_text), 1)
        for gone in ("Outcome:", "Summary:", "answered"):
            self.assertNotIn(gone, exp_text[0])
        from memorium.experience import POINTER_SCOPE_LINE
        self.assertIn(POINTER_SCOPE_LINE + "\nEvidence (verbatim records):\n[D1:2]", exp_text[0])
        self.assertIn(f"Task: {Q}", exp_text[0])
        self.assertFalse([v for v in self.sent() if v.kind == "V3"])          # generated picture removed
        self.assertNotIn(first, adopted)                                      # pointer is never adoption
        self.assertEqual(s.records[first].adopted, 0)
        self.assertEqual(s.records[first].pointer_cited, 1)
        self.assertEqual(s.last_pointer_cited, [first])

    def test_same_picture_keeps_the_full_node(self):
        s = self.store(admission=True)
        _, first, _ = self.ask(s, 0, picture="red")
        r, _, adopted = self.ask(s, 1, picture="red")
        self.assertEqual(r.experience_admission["admission"]["decision"], "full")
        self.assertEqual(r.experience_admission["admission"]["reason"], "same picture")
        node = self.unit.memory.nodes[first]
        self.assertIn(node.view("T0"), self.sent())
        self.assertIn(node.view("V3"), self.sent())
        self.assertIn(first, adopted)

    def test_question_without_picture_keeps_the_full_node(self):
        s = self.store(admission=True)
        _, first, _ = self.ask(s, 0, picture="red")
        r, _, _ = self.ask(s, 1, picture=None)
        self.assertEqual(r.experience_admission["admission"]["decision"], "full")
        self.assertEqual(r.experience_admission["admission"]["reason"], "question shows no picture")
        self.assertIn(self.unit.memory.nodes[first].view("T0"), self.sent())

    def test_source_without_picture_is_a_pointer_for_a_pictured_question(self):
        s = self.store(admission=True)
        _, first, _ = self.ask(s, 0, picture=None)
        self.assertIsNone(self.unit.memory.nodes[first].canvas["query_image_sha256"])
        r, _, _ = self.ask(s, 1, picture="red")
        self.assertEqual(r.experience_admission["admission"]["decision"], "pointer")
        self.assertEqual(r.experience_admission["admission"]["reason"], "source task showed no picture")

    def test_older_node_without_recorded_picture(self):
        old = self.store(admission=False)
        _, first, _ = self.ask(old, 0, picture="red")
        self.assertNotIn("query_image_sha256", self.unit.memory.nodes[first].canvas)  # default: canvas unchanged
        s = self.store(admission=True)
        s.records, s.index = old.records, old.index          # same nodes, now read with admission on
        r, _, _ = self.ask(s, 1, picture="red", record=False)
        self.assertEqual(r.experience_admission["admission"]["decision"], "pointer")
        self.assertEqual(r.experience_admission["admission"]["reason"], "source picture not recorded (older node)")
        self.assertFalse(r.experience_admission["admission"]["source_image_recorded"])
        r, _, _ = self.ask(s, 2, picture=None, record=False)
        self.assertEqual(r.experience_admission["admission"]["decision"], "full")

    def test_no_valid_source_pointer_skips_the_node(self):
        s = self.store(admission=True)
        _, first, _ = self.ask(s, 0, picture="red")
        node = self.unit.memory.nodes[first]
        saved = dict(self.unit.memory._sources)
        self.addCleanup(lambda: (self.unit.memory._sources.clear(), self.unit.memory._sources.update(saved)))
        for ref in node.source_refs:
            self.unit.memory._sources.pop(ref)
        entry, views = s.admit(first, None)
        self.assertEqual((entry["decision"], entry["reason"], views), ("skip", "no valid source pointer", []))

    def test_no_evidence_line_skips_the_pointer_and_sends_nothing(self):
        s = self.store(admission=True, node_tokens=40)    # too small for any verbatim evidence line
        _, first, _ = self.ask(s, 0, picture="red")
        self.assertTrue(self.unit.memory.nodes[first].view("T0").content.endswith("Evidence (verbatim records):"))
        r, _, _ = self.ask(s, 1, picture="blue")
        self.assertEqual(r.experience_admission["admission"]["decision"], "skip")
        self.assertNotIn(first, r.selected)
        self.assertFalse([v for v in self.sent() if v.content.startswith("Experience from")])

    def test_staged_and_long_term_candidates_are_both_ranked_and_logged(self):
        s = self.store(admission=True, top_k=3, image="off", promote_uses=1, promote_families=1)
        _, first, _ = self.ask(s, 0, picture=None)
        self.ask(s, 1, picture=None)                        # adopts the first node (full: no picture)
        self.assertEqual(s.consolidate(2), [first])
        self.assertEqual(s.records[first].state, "long_term")
        r, _, _ = self.ask(s, 3, picture="red", record=False)
        states = {c["node"]: c["state"] for c in r.experience_admission["candidates"]}
        self.assertEqual(states[first], "long_term")
        self.assertIn("staged", states.values())
        self.assertEqual([c["rank"] for c in r.experience_admission["candidates"]], list(range(len(states))))

    def test_pointer_reads_never_promote_or_draw(self):
        s = self.store(admission=True, image="adopt", promote_uses=1, promote_families=1)
        _, first, _ = self.ask(s, 0, picture="red")
        for step in range(1, 4):
            self.ask(s, step, picture="blue", record=False)
        self.assertEqual(s.records[first].adopted, 0)
        self.assertEqual(self.drawn, [])                      # no picture on a pointer "use"
        self.assertEqual(s.consolidate(4), [])
        self.assertEqual(s.summary()["admitted"], {"pointer": 3})

    # ---- historical behaviour -------------------------------------------------------------------------
    def test_default_off_sends_the_whole_node_whatever_the_picture(self):
        s = self.store()
        _, first, _ = self.ask(s, 0, picture="red")
        r, _, adopted = self.ask(s, 1, picture="blue")
        node = self.unit.memory.nodes[first]
        self.assertIsNone(r.experience_admission)
        self.assertEqual([v for v in self.sent() if v.kind in ("T0", "V3") and v.content.startswith(("Experience", "/tmp"))],
                         [node.view("T0"), node.view("V3")])
        self.assertIn(first, adopted)                         # the old (copied) answer counts as adopted

    def test_admission_needs_the_outside_placement(self):
        with self.assertRaises(ValueError):
            self.store(admission=True, outside=False)


class PointerTextTest(unittest.TestCase):
    """No numpy needed."""

    BLOCK = ("Experience from an earlier task (task 4)\nTask: What plant is shown?\nSearch: routes TV; 9 candidates; "
             "read 3 nodes\nOutcome: answered \"foxglove\"; the quoted records support it\nSummary: foxglove [D8:1].\n"
             "Evidence (verbatim records):\n[D8:1] Maya (1 May 2024): Maya: foxgloves!\nAssistant: Lovely.")

    def setUp(self):
        try:
            from memorium.experience import pointer_text  # noqa: F401
        except ImportError as exc:
            self.skipTest(str(exc))

    def test_pointer_text_without_evidence_is_none(self):
        from memorium.experience import pointer_text
        self.assertIsNone(pointer_text("Experience from an earlier task (task 1)\nTask: x\nOutcome: answered \"y\""))
        self.assertIsNone(pointer_text("Experience from an earlier task (task 1)\nTask: x\nEvidence (verbatim records):"))


if __name__ == "__main__":
    unittest.main()
