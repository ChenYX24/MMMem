from datetime import date
from pathlib import Path
import unittest

from memorium import Memory, MemoryPatch, SourceRecord, TaskSpec
from memorium.adapters.route_index import HashEncoder, RouteIndex
from memorium.core import View
from memorium.structure import annotate_relative_time, candidate_entities, segment_session, Turn
from memorium.u05 import ArmConfig, build_indices, parse_reader, run_task
from memorium.unit import UnitConfig, attach_view, build_unit, note_id
from memorium.views import MaterializationPolicy, ViewConfig, keeps_v2
from memorium.views import node_kind


def turn(index, session, speaker, text, when="1:56 pm on 8 May, 2023"):
    return SourceRecord(f"locomo:conv-x:D{session}:{index}", index, text,
                        {"dataset": "locomo", "sample_id": "conv-x", "session_index": session,
                         "dia_id": f"D{session}:{index}", "speaker": speaker, "date_time": when})


def unit_sources():
    return [
        turn(1, 1, "Caroline", "I went to a support group yesterday and it was powerful."),
        turn(2, 1, "Melanie", "That sounds great. My snakes Susie and Seraphim are doing well."),
        turn(3, 1, "Caroline", "I love that you named them Susie and Seraphim!"),
        turn(4, 2, "Melanie", "I ran a charity race last Saturday.", "9:00 am on 25 May, 2023"),
        turn(5, 2, "Caroline", "Congrats! Did Susie come along?", "9:00 am on 25 May, 2023"),
    ]


class Reader:
    """Answers from the first rendered record and cites it; records what it was shown."""

    def __init__(self, reply=None):
        self.seen = []
        self.reply = reply

    def __call__(self, prompt, views):
        self.seen.append([view.content for view in views])
        if self.reply is not None:
            return {"output": self.reply.pop(0) if isinstance(self.reply, list) else self.reply}
        first = views[0].content.splitlines()[0] if views else ""
        ident = first.split("]")[0].lstrip("[") if first.startswith("[") else ""
        return {"output": f"Answer: {first[:40]}\nEvidence: {ident}"}


def build(**overrides):
    unit = build_unit("conv-x", unit_sources(), UnitConfig(**overrides))
    index = RouteIndex(HashEncoder())
    build_indices(unit, index, None)
    return unit, index


class TimeTest(unittest.TestCase):
    def test_relative_times_keep_words_and_gain_absolute_dates(self):
        anchor = date(2023, 5, 8)
        self.assertEqual(annotate_relative_time("I went yesterday.", anchor), "I went yesterday [= 7 May 2023].")
        self.assertIn("[= June 2023]", annotate_relative_time("camping next month", anchor))
        self.assertIn("[= the week before 8 May 2023]", annotate_relative_time("an event last week", anchor))
        self.assertIn("[= Saturday, 6 May 2023]", annotate_relative_time("last Saturday", anchor))
        self.assertEqual(annotate_relative_time("no time here", anchor), "no time here")
        self.assertEqual(annotate_relative_time("yesterday", None), "yesterday")


class StructureTest(unittest.TestCase):
    def test_segments_never_split_or_drop_a_turn(self):
        turns = [Turn(f"t{i}", i, "s", "word " * (50 + i)) for i in range(20)]
        segments = segment_session(turns, max_tokens=200)
        self.assertEqual([t.id for segment in segments for t in segment], [t.id for t in turns])

    def test_entities_skip_speakers_and_sentence_initial_words(self):
        turns = [Turn("a", 0, "", "Thanks! My snake Susie is fine."), Turn("b", 1, "", "Yes, Susie loves Paris.")]
        found = candidate_entities(turns, speakers={"Melanie"}, min_mentions=2)
        self.assertEqual(list(found), ["Susie"])

    def test_unit_builds_segments_sessions_entities_and_timeline(self):
        unit, _ = build()
        kinds = {node_kind(node) for node in unit.memory.nodes.values()}
        self.assertEqual(kinds, {"segment", "session", "entity", "timeline"})
        timeline = next(n for n in unit.memory.nodes.values() if node_kind(n) == "timeline")
        self.assertIn("[= 7 May 2023]", timeline.view("T0").content)
        session_two = unit.memory.nodes["conv-x:session:2"]
        self.assertEqual(session_two.parent, "conv-x:session:1")


class ViewTest(unittest.TestCase):
    def test_result_records_every_view_with_its_owning_node(self):
        from memorium.unit import text_of

        unit, _ = build()
        node_id = unit.segments[0]
        node = unit.memory.nodes[node_id]
        generated = View("V2", "/tmp/generated.png", node.source_refs, "qwen_image", 50)
        attach_view(unit.memory, node_id, generated, "generated-trace-test")
        index = RouteIndex(HashEncoder())
        build_indices(unit, index, None)
        result = run_task(unit, TaskSpec(text_of(unit.memory.nodes[node_id])), "q",
                          ArmConfig(routes=("T",), top_k=1, hops=0, token_budget=1000,
                                    read_generated_views=True),
                          text_index=index, visual_index=None,
                          reader=lambda prompt, views: {"output": "Answer: yes\nEvidence: D1:1"})
        self.assertEqual(result.selected, [node_id])
        self.assertEqual(result.views, [f"{node_id}:T0", f"{node_id}:V2"])

    def test_generated_views_reach_the_reader_only_when_enabled_and_within_budget(self):
        from memorium.u05 import select

        unit, _ = build()
        node_id = unit.segments[0]
        view = View("V2", "/tmp/generated.png", unit.memory.nodes[node_id].source_refs, "qwen_image", 50)
        attach_view(unit.memory, node_id, view, "generated-view-test")
        for enabled, budget, expected in ((False, 1000, False), (True, 1000, True), (True, 1, False)):
            selected = select(unit, "q", [node_id],
                              ArmConfig(read_generated_views=enabled, token_budget=budget), None)
            self.assertEqual(any(item.kind == "V2" for item in selected.views), expected)

    def test_views_are_optional_versioned_and_replayable(self):
        unit, _ = build()
        segment = unit.segments[0]
        before = unit.memory.revision
        view = View("V1", "/tmp/page.png", unit.memory.nodes[segment].source_refs, "canvas", 300)
        attach_view(unit.memory, segment, view, "upgrade:v1")
        self.assertEqual(unit.memory.nodes[segment].view("V1"), view)
        self.assertEqual(unit.memory.revision, before + 1)
        replayed = Memory.replay(unit.memory.sources.values(), unit.memory._patches)
        self.assertEqual(replayed.snapshot_hash, unit.memory.snapshot_hash)

    def test_a_view_cannot_cite_sources_outside_its_node(self):
        unit, _ = build()
        node = unit.memory.nodes[unit.segments[0]]
        with self.assertRaises(ValueError):
            attach_view(unit.memory, node.id, View("T1", "x", ("locomo:conv-x:D2:5",), "w", 1), "bad")

    def test_upgrades_wait_for_repeated_use_and_respect_the_image_budget(self):
        unit, _ = build()
        policy = MaterializationPolicy(ViewConfig(t1=True, v2=True, upgrade_after_uses=2, v2_unit_budget=0))
        node = unit.memory.nodes[unit.segments[0]]
        unit.ledger.use(node.id)
        self.assertEqual(policy.upgrades(node, unit.ledger), ())
        unit.ledger.use(node.id)
        self.assertEqual(policy.upgrades(node, unit.ledger), ("T1",))

    def test_v2_coverage_subsample_is_deterministic(self):
        picks = [keeps_v2(f"n{i}", 0.5) for i in range(400)]
        self.assertEqual(picks, [keeps_v2(f"n{i}", 0.5) for i in range(400)])
        self.assertTrue(120 < sum(picks) < 280)


class LoopTest(unittest.TestCase):
    def test_note_written_after_reading_is_retrieved_by_a_later_task(self):
        unit, index = build()
        reader = Reader()
        arm = ArmConfig(routes=("T",), token_budget=2000)
        first = run_task(unit, TaskSpec("What are the snakes named?"), "q1", arm, text_index=index,
                         visual_index=None, reader=reader)
        self.assertIsNotNone(first.note)
        self.assertEqual(unit.memory.nodes[first.note].canvas["type"], "note")
        second = run_task(unit, TaskSpec("Which snakes does Melanie have?"), "q2", arm, text_index=index,
                          visual_index=None, reader=reader)
        self.assertIn(first.note, second.retrieved)

    def test_repeated_question_versions_the_same_note(self):
        unit, index = build()
        arm = ArmConfig(routes=("T",))
        run_task(unit, TaskSpec("What are the snakes named?"), "q1", arm, text_index=index,
                 visual_index=None, reader=Reader())
        version = unit.memory.nodes[note_id("conv-x", "What are the snakes named?")].version
        again = run_task(unit, TaskSpec("What are the snakes named?"), "q2", arm, text_index=index,
                         visual_index=None, reader=Reader())
        self.assertTrue(again.note_updated)
        self.assertGreater(unit.memory.nodes[again.note].version, version)

    def test_write_back_off_hides_notes(self):
        unit, index = build()
        run_task(unit, TaskSpec("What are the snakes named?"), "q1", ArmConfig(routes=("T",)),
                 text_index=index, visual_index=None, reader=Reader())
        off = run_task(unit, TaskSpec("Which snakes does Melanie have?"), "q2",
                       ArmConfig(routes=("T",), write_back=False), text_index=index, visual_index=None,
                       reader=Reader())
        self.assertFalse(any(node_kind(unit.memory.nodes[n]) == "note" for n in off.retrieved))
        self.assertIsNone(off.note)

    def test_notes_hold_only_model_output(self):
        unit, index = build()
        result = run_task(unit, TaskSpec("What are the snakes named?"), "q1", ArmConfig(routes=("T",)),
                          text_index=index, visual_index=None,
                          reader=Reader("Answer: Susie and Seraphim\nEvidence: D1:2"))
        text = unit.memory.nodes[result.note].view("T0").content
        self.assertIn("Susie and Seraphim", text)
        self.assertIn("unverified", text)
        self.assertEqual(result.cited, ["locomo:conv-x:D1:2"])

    def test_a_selected_note_brings_the_segment_holding_its_evidence(self):
        unit, index = build()
        first = run_task(unit, TaskSpec("What are the snakes named?"), "q1", ArmConfig(routes=("T",)),
                         text_index=index, visual_index=None,
                         reader=Reader("Answer: Susie and Seraphim\nEvidence: D1:2"))
        holder = next(n for n, node in unit.memory.nodes.items()
                      if node_kind(node) == "segment" and "locomo:conv-x:D1:2" in node.source_refs)
        self.assertEqual(unit.memory.nodes[first.note].related[0], holder)
        second = run_task(unit, TaskSpec("snakes named"), "q2", ArmConfig(routes=("T",), token_budget=5000),
                          text_index=index, visual_index=None, reader=Reader())
        self.assertIn(first.note, second.selected)
        self.assertIn(holder, second.selected)

    def test_insufficient_selects_an_offered_no_information_option(self):
        unit, index = build()
        reader = Reader("Answer: insufficient\nNeed: x")
        task = TaskSpec("Did Bo buy a boat?", ("Bo bought a boat.", "No information available"))
        result = run_task(unit, task, "q", ArmConfig(routes=("T",)), text_index=index, visual_index=None,
                          reader=reader)
        self.assertEqual(result.answer, "No information available")
        self.assertIsNone(result.note)
        plain = run_task(unit, TaskSpec("Did Bo buy a boat?"), "q2", ArmConfig(routes=("T",)),
                         text_index=index, visual_index=None, reader=Reader("Answer: insufficient\nNeed: x"))
        self.assertEqual(plain.answer, "")

    def test_note_controls_limit_notes_and_stop_pulling_segments(self):
        unit, index = build()
        reply = "Answer: Susie and Seraphim\nEvidence: D1:2"
        for question in ("What are the snakes named?", "Name the snakes?", "Which snakes?"):
            run_task(unit, TaskSpec(question), question, ArmConfig(routes=("T",)), text_index=index,
                     visual_index=None, reader=Reader(reply))
        arm = ArmConfig(routes=("T",), token_budget=5000, note_limit=1, note_pull="none")
        result = run_task(unit, TaskSpec("snakes named"), "q", arm, text_index=index, visual_index=None,
                          reader=Reader(reply))
        self.assertEqual(sum(node_kind(unit.memory.nodes[n]) == "note" for n in result.selected), 1)

    def test_similarity_gate_drops_unrelated_notes(self):
        unit, index = build()
        run_task(unit, TaskSpec("What are the snakes named?"), "q1", ArmConfig(routes=("T",)), text_index=index,
                 visual_index=None, reader=Reader("Answer: Susie\nEvidence: D1:2"))
        gated = run_task(unit, TaskSpec("When was the charity race?"), "q2",
                         ArmConfig(routes=("T",), note_min_sim=0.99), text_index=index, visual_index=None,
                         reader=Reader())
        self.assertFalse(any(node_kind(unit.memory.nodes[n]) == "note" for n in gated.retrieved))

    def test_evidence_style_note_carries_its_cited_records(self):
        unit, index = build()
        result = run_task(unit, TaskSpec("What are the snakes named?"), "q", ArmConfig(routes=("T",), note_style="evidence"),
                          text_index=index, visual_index=None, reader=Reader("Answer: Susie\nEvidence: D1:2"))
        self.assertIn("My snakes Susie and Seraphim", unit.memory.nodes[result.note].view("T0").content)

    def test_iterative_retrieval_runs_at_most_one_more_round(self):
        unit, index = build()
        reader = Reader(["Answer: insufficient\nNeed: charity race", "Answer: insufficient\nNeed: x"])
        result = run_task(unit, TaskSpec("When was the race?"), "q", ArmConfig(routes=("T",), iterative=True),
                          text_index=index, visual_index=None, reader=reader)
        self.assertEqual(result.iterations, 2)
        self.assertEqual(result.answer, "")
        self.assertIsNone(result.note)

    def test_selection_stays_within_the_token_budget(self):
        unit, index = build()
        result = run_task(unit, TaskSpec("snakes"), "q", ArmConfig(routes=("T",), token_budget=60),
                          text_index=index, visual_index=None, reader=Reader())
        self.assertLessEqual(result.text_tokens + result.image_tokens, 60)

    def test_parse_reader_formats(self):
        self.assertEqual(parse_reader("Answer: 7 May 2023\nEvidence: D1:3, D1:4"), ("7 May 2023", ["D1:3", "D1:4"], None))
        self.assertEqual(parse_reader("Answer: insufficient\nNeed: the date"), ("", [], "the date"))
        self.assertEqual(parse_reader("Answer:\nEvidence: D1:3"), ("", ["D1:3"], None))


if __name__ == "__main__":
    unittest.main()


class IntegrationCheckTest(unittest.TestCase):
    def test_code_compares_observed_people_and_setting_with_the_spec(self):
        from memorium.adapters.imagegen import IntegrationSpec, parse_check, parse_spec

        spec = parse_spec('{"facts":[{"text":"Jon opened a studio","source_id":"D1:1"}],"entities":["Jon"],'
                          '"scene":"Jon dances alone in a studio","people":1,"setting":"dance studio"}', ["D1:1"])
        self.assertEqual((spec.people, spec.setting), (1, "dance studio"))
        self.assertTrue(parse_check('{"people": 1, "setting": "studio", "objects": ["mirror"]}', spec)[0])
        self.assertFalse(parse_check('{"people": 3, "setting": "studio"}', spec)[0])
        self.assertFalse(parse_check('{"people": 1, "setting": "swimming pool"}', spec)[0])
        self.assertFalse(parse_check("not json", spec)[0])
        crowd = IntegrationSpec(spec.facts, (), "a crew performs", 5, "stage")
        self.assertTrue(parse_check('{"people": 4, "setting": "concert stage"}', crowd)[0])

    def test_gender_is_checked_for_small_scenes(self):
        from memorium.adapters.imagegen import IntegrationSpec, parse_check

        spec = IntegrationSpec((("x", "D1:1"),), ("Jon",), "a man in a studio", 1, "studio", ("man",))
        self.assertTrue(parse_check('{"people":1,"genders":["man"],"setting":"studio"}', spec)[0])
        self.assertFalse(parse_check('{"people":1,"genders":["woman"],"setting":"studio"}', spec)[0])
        pair = IntegrationSpec(spec.facts, (), "a man and a woman", 2, "studio", ("man", "woman"))
        self.assertTrue(parse_check('{"people":2,"genders":["woman","man"],"setting":"studio"}', pair)[0])
        self.assertFalse(parse_check('{"people":2,"genders":["woman","woman"],"setting":"studio"}', pair)[0])
        unknown = IntegrationSpec(spec.facts, (), "someone", 1, "studio", ("unknown",))
        self.assertTrue(parse_check('{"people":1,"genders":["woman"],"setting":"studio"}', unknown)[0])

    def test_facts_citing_foreign_sources_are_dropped(self):
        from memorium.adapters.imagegen import parse_spec

        with self.assertRaises(ValueError):
            parse_spec('{"facts":[{"text":"x","source_id":"D9:9"}],"scene":"s"}', ["D1:1"])


class NativeImageTest(unittest.TestCase):
    def gallery_unit(self, tmp):
        from PIL import Image
        from memorium.native import attach_native_views

        paths = []
        for index, size in enumerate(((2000, 1500), (300, 300))):
            path = f"{tmp}/img{index}.jpg"
            Image.new("RGB", size, (index * 90, 40, 40)).save(path)
            paths.append(path)
        sources = [SourceRecord("mg:u:D1:1", 0, "hi", {"dataset": "memgallery", "sample_id": "u", "session_index": "D1",
                                                       "dia_id": "D1:1", "speaker": "Ann", "date_time": "2024-06-17",
                                                       "image_ids": ["D1:IMG_001", "D1:IMG_002"]}, tuple(paths)),
                   SourceRecord("mg:u:D1:2", 1, "a robot", {"dataset": "memgallery", "sample_id": "u",
                                                            "session_index": "D1", "dia_id": "D1:2", "speaker": "Ann",
                                                            "date_time": "2024-06-17"})]
        unit = build_unit("u", sources, UnitConfig(structure=False))
        added = attach_native_views(unit, directory=Path(tmp) / "v0", count_image=lambda p: 100, max_side=640)
        return unit, added

    def test_native_images_are_bounded_and_travel_with_their_segment(self):
        import tempfile
        from PIL import Image

        with tempfile.TemporaryDirectory() as tmp:
            unit, added = self.gallery_unit(tmp)
            self.assertEqual(added, 2)
            segment = unit.memory.nodes[unit.segments[0]]
            v0 = [v for v in segment.views if v.kind == "V0"]
            self.assertEqual([v.metadata["image_id"] for v in v0], ["D1:IMG_001", "D1:IMG_002"])
            with Image.open(v0[0].content) as image:
                self.assertLessEqual(max(image.size), 640)
            index = RouteIndex(HashEncoder())
            build_indices(unit, index, None)
            reader = Reader()
            result = run_task(unit, TaskSpec("robot?"), "q", ArmConfig(routes=("T",), write_back=False),
                              text_index=index, visual_index=None, reader=reader)
            self.assertEqual(result.image_tokens, 200)
            tight = run_task(unit, TaskSpec("robot?"), "q2", ArmConfig(routes=("T",), write_back=False,
                                                                       token_budget=result.text_tokens + 150),
                             text_index=index, visual_index=None, reader=reader)
            self.assertLessEqual(tight.text_tokens + tight.image_tokens, result.text_tokens + 150)

    def test_insufficient_becomes_the_stated_abstention_reply(self):
        unit, index = build()
        rule = "Only if the information is not present, reply with: \u201cNot mentioned.\u201d"
        result = run_task(unit, TaskSpec("Did he mention Mars?", constraints=(rule,)), "q",
                          ArmConfig(routes=("T",)), text_index=index, visual_index=None,
                          reader=Reader("Answer: insufficient\nNeed: x"))
        self.assertEqual(result.answer, "Not mentioned.")
        self.assertIsNone(result.note)
        plain = run_task(unit, TaskSpec("Did he mention Mars?"), "q2", ArmConfig(routes=("T",)), text_index=index,
                         visual_index=None, reader=Reader("Answer: insufficient\nNeed: x"))
        self.assertEqual(plain.answer, "")

    def test_task_constraints_reach_the_reader_prompt(self):
        unit, index = build()
        prompts = []

        def reader(prompt, views):
            prompts.append(prompt)
            return {"output": "Answer: Yes.\nEvidence:"}

        run_task(unit, TaskSpec("Any conflict?", constraints=("Reply strictly with Yes. or No.",)), "q",
                 ArmConfig(routes=("T",), write_back=False), text_index=index, visual_index=None, reader=reader)
        self.assertIn("Reply strictly with Yes. or No.", prompts[0])


class SessionPageTest(unittest.TestCase):
    def test_a_session_page_is_indexed_and_expands_to_its_segments(self):
        from memorium.u05 import expand

        unit, _ = build()
        session = "conv-x:session:1"
        attach_view(unit.memory, session, View("V1", "/tmp/session_page.png", unit.memory.nodes[session].source_refs,
                                               "canvas", 500), "page")
        text_index, visual_index = RouteIndex(HashEncoder()), RouteIndex(HashEncoder())
        build_indices(unit, text_index, visual_index)
        self.assertEqual(visual_index.rank("session_page", "conv-x", top_k=5), [session])
        self.assertEqual(text_index.rank("session", "conv-x", top_k=50).count(session), 0)
        children = unit.memory.nodes[session].canvas["children"]
        self.assertTrue(set(children) <= set(expand(unit, [session], 1)))
