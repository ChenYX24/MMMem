import unittest


class ExperienceTest(unittest.TestCase):
    """Built unit + hashing encoder + scripted reader (numpy needed: runs in the hospital environment)."""

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

    def store(self, **kw):
        from memorium.experience import ExperienceStore, Settings
        drawn = []
        s = ExperienceStore(self.unit, self.RouteIndex(self.HashEncoder()), Settings(**kw), text_index=self.index,
                            summarize=lambda p: "Melanie's snakes are Susie and Seraphim [D1:2].",
                            draw=lambda scene, seed: drawn.append(scene) or f"/tmp/exp{seed}.png",
                            count_image=lambda path: 100)
        return s, drawn

    def reader(self, prompt, views):
        text = "\n".join(v.content for v in views)
        if "snake" in prompt.lower() and "Susie" in text:
            return {"output": "Answer: Susie and Seraphim\nEvidence: D1:2", "status": "OK"}
        return {"output": "Answer: insufficient\nNeed: more", "status": "OK"}

    def ask(self, store, q, step):
        from memorium import TaskSpec
        from memorium.u05 import ArmConfig, run_task
        arm = ArmConfig(routes=("T",), write_back=False, token_budget=3000)
        r = run_task(self.unit, TaskSpec(q), f"q{step}", arm, text_index=self.index, visual_index=None,
                     reader=self.reader, staging=store, step=step)
        adopted = store.credit(q, r.answer, r.selected, r.cited, step)
        node = store.record(q, r.answer, r, step, "T")
        return r, node, adopted

    def test_every_task_writes_a_traceable_experience_node(self):
        s, _ = self.store(image="off")
        r, node, _ = self.ask(s, "What are Melanie's snakes called?", 0)
        n = self.unit.memory.nodes[node]
        self.assertEqual(n.canvas["type"], "experience")
        self.assertTrue(n.canvas["supported"])
        self.assertIn("locomo:conv-x:D1:2", n.source_refs)
        t0 = n.view("T0").content
        self.assertIn("Summary:", t0)
        self.assertIn("[D1:2] Melanie", t0)
        self.assertIn("Search: routes T", t0)
        # an abstained task still leaves an experience, marked unsupported
        _, node2, _ = self.ask(s, "Where did Caroline go yesterday?", 1)
        self.assertFalse(self.unit.memory.nodes[node2].canvas["supported"])

    def test_outside_placement_does_not_spend_the_evidence_budget(self):
        s, _ = self.store(image="off", outside=True)
        self.ask(s, "What are Melanie's snakes called?", 0)
        from memorium import TaskSpec
        from memorium.u05 import ArmConfig, run_task
        arm = ArmConfig(routes=("T",), write_back=False, token_budget=3000)
        seen = {}

        def reader(prompt, views):
            seen["views"] = list(views)
            return self.reader(prompt, views)
        r = run_task(self.unit, TaskSpec("What are Melanie's snakes called?"), "q1", arm, text_index=self.index,
                     visual_index=None, reader=reader, staging=s, step=1)
        exp = [n for n in r.selected if n.startswith("conv-x:experience:")]
        self.assertEqual(len(exp), 1)
        # the budgeted ranking holds only source nodes; the experience node comes after them
        self.assertEqual(r.selected[-1], exp[0])
        self.assertIn(self.unit.memory.nodes[exp[0]].view("T0"), seen["views"])
        self.assertLessEqual(r.text_tokens - self.unit.memory.nodes[exp[0]].view("T0").token_cost, 3000)

    def test_context_overflow_halves_the_evidence_budget_and_is_reported(self):
        from memorium import TaskSpec
        from memorium.u05 import ArmConfig, run_task
        arm = ArmConfig(routes=("T",), write_back=False, token_budget=3000)
        seen = []

        def reader(prompt, views):
            seen.append(sum(v.token_cost for v in views))
            if len(seen) <= 2:
                raise ValueError("model context cap exceeded; no silent truncation")
            return self.reader(prompt, views)
        r = run_task(self.unit, TaskSpec("What are Melanie's snakes called?"), "q", arm, text_index=self.index,
                     visual_index=None, reader=reader)
        self.assertEqual(r.budget_halvings, 2)
        self.assertEqual(len(seen), 3)
        self.assertIn("Susie", r.answer)

    def test_other_errors_and_a_fitting_call_are_untouched(self):
        from memorium import TaskSpec
        from memorium.u05 import ArmConfig, run_task
        arm = ArmConfig(routes=("T",), write_back=False, token_budget=3000)
        ok = run_task(self.unit, TaskSpec("What are Melanie's snakes called?"), "q", arm, text_index=self.index,
                      visual_index=None, reader=self.reader)
        self.assertEqual(ok.budget_halvings, 0)

        def broken(prompt, views):
            raise ValueError("something else")
        with self.assertRaises(ValueError):
            run_task(self.unit, TaskSpec("What are Melanie's snakes called?"), "q", arm, text_index=self.index,
                     visual_index=None, reader=broken)

    def test_staged_first_adoption_and_promotion(self):
        s, drawn = self.store(promote_uses=2, promote_families=2, image="adopt")
        _, first, _ = self.ask(s, "What are Melanie's snakes called?", 0)
        self.assertEqual(s.records[first].state, "staged")
        self.assertNotIn(first, {o for o in self.index._owners})       # not in the main index while staged
        r, _, adopted = self.ask(s, "Which snakes does Melanie have, Susie?", 1)
        self.assertIn(first, r.retrieved)                                # staged route searched first
        self.assertIn(first, adopted)
        self.assertEqual(len(drawn), 1)                                  # picture on first adoption
        self.assertIsNotNone(self.unit.memory.nodes[first].view("V3"))
        self.ask(s, "Who are Susie and Seraphim, Melanie's snakes?", 2)
        self.assertEqual(s.consolidate(3), [first])
        self.assertEqual(s.records[first].state, "long_term")
        self.assertIn(first, set(self.index._owners))                    # now retrieved like any node

    def test_rephrasings_are_capped_and_ablations(self):
        s, _ = self.store(promote_uses=3, promote_families=1, family_cap=1, image="off")
        _, first, _ = self.ask(s, "What are Melanie's snakes called?", 0)
        for step in range(1, 5):
            self.ask(s, "What are Melanie's snakes called?", step)
        self.assertEqual(s.records[first].adopted, 1)
        s2, _ = self.store(staging=False, image="off")
        _, n, _ = self.ask(s2, "What are Melanie's snakes called?", 0)
        self.assertEqual(s2.records[n].state, "long_term")


if __name__ == "__main__":
    unittest.main()


class WarmupTaskTest(unittest.TestCase):
    def test_warmup_rows_become_unscored_tasks_of_their_unit(self):
        from memorium.experience import warmup_tasks
        rows = [{"unit": "a", "question": "Q1?", "warmup": True, "answer": "secret"},
                {"unit": "b", "question": "Q2?", "warmup": True, "options": ["x", "y"]},
                {"unit": "a", "question": "Q3?", "warmup": True}]
        tasks = warmup_tasks(rows, "a")
        self.assertEqual([t["question"] for t in tasks], ["Q1?", "Q3?"])
        self.assertTrue(all(t["warmup"] and t["question_index"] == -1 for t in tasks))
        self.assertNotIn("answer", tasks[0])
        self.assertEqual(len({t["id"] for t in tasks}), 2)

    def test_rows_not_marked_as_warmup_are_refused(self):
        from memorium.experience import warmup_tasks
        with self.assertRaises(ValueError):
            warmup_tasks([{"unit": "a", "question": "Q?"}], "a")


class WarmupPromptTest(unittest.TestCase):
    def test_warmup_task_can_fill_the_loogle_prompt_without_official_fields(self):
        from memorium.experience import warmup_tasks
        task = warmup_tasks([{"unit": "a", "question": "Which year?\n(A) 1\n(B) 2\n(C) 3\n(D) 4", "warmup": True}], "a")[0]
        for key in ("instruction", "loogle_question", "loogle_options"):
            self.assertIn(key, task)
        try:
            from memorium.adapters.loogle import official_prompt
        except ImportError as exc:
            self.skipTest(str(exc))
        text = official_prompt({"instruction": task["instruction"], "question": task["loogle_question"],
                                "options": task["loogle_options"]}, "CTX")
        self.assertIn("CTX", text)
        self.assertIn("Which year?", text)
        self.assertNotIn("{", text.replace("(A)", ""))
