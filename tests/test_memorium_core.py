import unittest
from concurrent.futures import ThreadPoolExecutor


class MemoriumCoreTest(unittest.TestCase):
    def test_source_is_content_addressed_and_task_rejects_answer_leakage(self):
        from memorium import SourceRecord, TaskSpec

        source = SourceRecord(
            id="doc-1:turn-2",
            ordinal=2,
            text="Ada approved the budget.",
            locator={"document_id": "doc-1", "turn": 2},
        )
        self.assertEqual(source.content_hash, source.recompute_hash())
        self.assertEqual(source.source_refs(), ("doc-1:turn-2",))

        task = TaskSpec(question="Who approved the budget?", options=("Ada", "Bo"))
        self.assertEqual(task.public_query, "Who approved the budget?\n(a) Ada\n(b) Bo")
        with self.assertRaises(ValueError):
            TaskSpec(question="Who approved?", metadata={"gold_answer": "Ada"})

    def test_source_and_task_copy_mutable_inputs(self):
        from memorium import SourceRecord, TaskSpec

        locator = {"document_id": "doc-1"}
        metadata = {"split": "dev"}
        source = SourceRecord("doc-1:turn-2", 2, "Ada approved the budget.", locator)
        task = TaskSpec("Who approved?", metadata=metadata)
        locator["document_id"] = "tampered"
        metadata["gold_answer"] = "Ada"

        self.assertEqual(source.locator["document_id"], "doc-1")
        self.assertEqual(source.content_hash, source.recompute_hash())
        self.assertNotIn("gold_answer", task.metadata)

    def test_patch_is_atomic_versioned_and_rejects_parent_cycle(self):
        from memorium import Memory, MemoryNode, MemoryPatch, SourceRecord

        source = SourceRecord("doc-1:turn-2", 2, "Ada approved the budget.", {"document_id": "doc-1"})
        memory = Memory((source,))
        root = MemoryNode("root", "Budget decision", (source.id,))
        receipt = memory.apply(MemoryPatch("p1", upsert=(root,)))
        self.assertEqual((receipt.revision, memory.nodes["root"].version), (1, 1))

        child = MemoryNode("child", "Approval detail", (source.id,), parent="root")
        memory.apply(MemoryPatch("p2", expected_revision=1, upsert=(child,)))
        self.assertEqual((memory.nodes["root"].version, memory.nodes["child"].version), (1, 2))
        before = memory.snapshot_hash
        with self.assertRaisesRegex(ValueError, "parent cycle"):
            memory.apply(MemoryPatch("bad", expected_revision=2, upsert=(
                MemoryNode("root", "Budget decision", (source.id,), parent="child"),
            )))
        self.assertEqual(memory.snapshot_hash, before)

    def test_candidate_union_keeps_dense_and_records_budget_drops(self):
        from memorium import Candidate, merge_candidates, select_candidates

        dense = Candidate("D", 1, source_refs=("doc-1:span-3",))
        text = Candidate("T", 1, node_id="root", source_refs=("doc-1:span-1",))
        hierarchy = Candidate("H", 2, node_id="child", source_refs=("doc-1:span-2",))
        merged = merge_candidates((dense,), (text,), (hierarchy,))

        self.assertEqual(merged.source_refs, ("doc-1:span-3", "doc-1:span-1", "doc-1:span-2"))
        selection = select_candidates(merged, limit=2)
        self.assertIn(dense, selection.selected)
        self.assertEqual(selection.dropped, (hierarchy,))

        explicit = select_candidates(merged, limit=1, selected=(hierarchy,))
        self.assertEqual(explicit.selected, (hierarchy,))
        self.assertEqual(explicit.dropped, (dense, text))
        with self.assertRaisesRegex(ValueError, "outside the merged union"):
            select_candidates(merged, selected=(Candidate("D", 2, ("other",)),))

    def test_each_retrieval_route_builds_candidates_from_versioned_memory(self):
        from memorium import Memory, MemoryNode, MemoryPatch, SourceRecord

        sources = (
            SourceRecord("s0", 0, "Ada approved the budget.", {"document_id": "doc-1"}),
            SourceRecord("s1", 1, "Bo recorded the decision.", {"document_id": "doc-1"}),
        )
        memory = Memory(sources)
        memory.apply(MemoryPatch("p1", upsert=(
            MemoryNode("root", "Budget decision", ("s0",),
                       canvas={"items": ({"kind": "text", "text": "Budget decision"},)}),
            MemoryNode("detail", "Decision record", ("s1",), parent="root"),
        )))

        self.assertEqual(memory.candidates("T", ("root",))[0].source_refs, ("s0",))
        self.assertEqual(memory.candidates("T", ("root",))[0].node_version, 1)
        self.assertEqual(memory.candidates("V", ("root",))[0].node_id, "root")
        self.assertEqual(memory.candidates("H", ("detail",))[0].route, "H")
        self.assertEqual(memory.candidates("D", ("s1",))[0].node_id, None)
        with self.assertRaisesRegex(ValueError, "visual representation"):
            memory.candidates("V", ("detail",))

    def test_hierarchy_navigation_is_local_and_deterministic(self):
        from memorium import Memory, MemoryNode, MemoryPatch, SourceRecord

        source = SourceRecord("s0", 0, "Observed fact", {"document_id": "doc-1"})
        memory = Memory((source,))
        memory.apply(MemoryPatch("p1", upsert=(
            MemoryNode("root", "Overview", ("s0",)),
            MemoryNode("left", "Left detail", ("s0",), parent="root", related=("right",)),
            MemoryNode("right", "Right detail", ("s0",), parent="root", related=("left",)),
        )))

        self.assertEqual(memory.navigate("left", "parent"), ("root",))
        self.assertEqual(memory.navigate("root", "children"), ("left", "right"))
        self.assertEqual(memory.navigate("left", "related"), ("right",))
        with self.assertRaisesRegex(ValueError, "navigation"):
            memory.navigate("left", "all")

    def test_source_view_keeps_speaker_and_timestamp_from_the_locator(self):
        from memorium import Memory, MemoryNode, MemoryPatch, SourceRecord, TaskSpec, merge_candidates

        dated = SourceRecord("locomo:conv-26:D1:1", 0, "Good to see you!",
                             {"dataset": "locomo", "speaker": "Caroline",
                              "date_time": "1:56 pm on 8 May, 2023"})
        plain = SourceRecord("doc-1:turn-2", 1, "Ada approved the budget.", {"document_id": "doc-1"})
        memory = Memory((dated, plain))
        memory.apply(MemoryPatch("p1", upsert=(
            MemoryNode("root", "Session", (dated.id, plain.id)),)))
        workspace = memory.workspace(
            TaskSpec("When did they meet?"),
            merge_candidates(memory.candidates("T", ("root",))),
        )

        # Without these a conversational benchmark cannot answer who or when.
        self.assertIn("[locomo:conv-26:D1:1] (Caroline @ 1:56 pm on 8 May, 2023)",
                      workspace.source_text_view)
        self.assertIn("[doc-1:turn-2]\nAda approved the budget.", workspace.source_text_view)

    def test_workspace_is_task_scoped_and_does_not_mutate_memory(self):
        from memorium import Candidate, Memory, MemoryNode, MemoryPatch, SourceRecord, TaskSpec, merge_candidates

        source = SourceRecord("doc-1:turn-2", 2, "Ada approved the budget.", {"document_id": "doc-1"})
        memory = Memory((source,))
        memory.apply(MemoryPatch("p1", upsert=(MemoryNode("root", "Budget decision", (source.id,)),)))
        workspace = memory.workspace(
            TaskSpec("Who approved the budget?"),
            merge_candidates(memory.candidates("T", ("root",))),
        )

        self.assertEqual([node.id for node in workspace.nodes], ["root"])
        self.assertEqual(workspace.source_refs, (source.id,))
        self.assertIn("Budget decision", workspace.text_view)
        self.assertEqual(workspace.sources, (source,))
        self.assertIn("Ada approved the budget.", workspace.source_text_view)
        self.assertEqual(memory.revision, 1)

        stale = memory.candidates("T", ("root",))
        memory.apply(MemoryPatch("p2", expected_revision=1,
                                 upsert=(MemoryNode("root", "Changed", (source.id,)),)))
        with self.assertRaisesRegex(ValueError, "version is stale"):
            memory.workspace(TaskSpec("Who approved?"), merge_candidates(stale))

    def test_retrieve_merges_routes_then_builds_selected_workspace(self):
        from memorium import Memory, MemoryNode, MemoryPatch, SourceRecord, TaskSpec

        sources = (
            SourceRecord("s0", 0, "Ada approved the budget.", {"document_id": "doc-1"}),
            SourceRecord("s1", 1, "Bo recorded the decision.", {"document_id": "doc-1"}),
        )
        memory = Memory(sources)
        memory.apply(MemoryPatch("p1", upsert=(
            MemoryNode("root", "Budget decision", ("s0",),
                       canvas={"items": ({"kind": "text", "text": "Budget decision"},)}),
            MemoryNode("detail", "Decision record", ("s1",), parent="root"),
        )))
        task = TaskSpec("Who recorded the decision?")
        full = memory.retrieve(task, {"T": ("root",), "V": ("root",),
                                      "H": ("detail",), "D": ("s1",)})
        chosen = (full.candidates.candidates[2], full.candidates.candidates[3])
        result = memory.retrieve(task, {"T": ("root",), "V": ("root",),
                                        "H": ("detail",), "D": ("s1",)},
                                 limit=2, selected=chosen)

        self.assertEqual([item.route for item in full.candidates.candidates], ["T", "V", "H", "D"])
        self.assertEqual(result.selection.selected, chosen)
        self.assertEqual([node.id for node in result.workspace.nodes], ["detail"])
        self.assertEqual(result.workspace.source_refs, ("s1",))

    def test_cost_ledger_deduplicates_calls_but_keeps_failures(self):
        from memorium import CostEvent, CostLedger

        ledger = CostLedger()
        event = CostEvent("reader", "qwen", "call-1", text_tokens=10, output_tokens=3,
                          cpu_seconds=0.2, gpu_seconds=1.5, wall_seconds=0.5, success=False)
        self.assertTrue(ledger.record(event))
        self.assertFalse(ledger.record(event))
        self.assertEqual(ledger.total.gpu_seconds, 1.5)
        self.assertEqual(ledger.events[0].success, False)

    def test_replay_is_deterministic_and_referenced_nodes_cannot_be_deleted(self):
        from memorium import Memory, MemoryNode, MemoryPatch, SourceRecord

        source = SourceRecord("doc-1:turn-2", 2, "Ada approved the budget.", {"document_id": "doc-1"})
        patches = (
            MemoryPatch("p1", upsert=(MemoryNode("root", "Budget decision", (source.id,)),)),
            MemoryPatch("p2", expected_revision=1,
                        upsert=(MemoryNode("child", "Approval detail", (source.id,), parent="root"),)),
        )
        memory = Memory.replay((source,), patches)
        self.assertEqual(memory.snapshot_hash, Memory.replay((source,), patches).snapshot_hash)
        before = memory.snapshot_hash
        with self.assertRaisesRegex(ValueError, "unknown parent reference"):
            memory.apply(MemoryPatch("bad-delete", expected_revision=2, delete=("root",)))
        self.assertEqual(memory.snapshot_hash, before)

    def test_commit_pins_and_verifies_source_content_versions(self):
        from memorium import Memory, MemoryNode, MemoryPatch, SourceRecord

        source = SourceRecord("doc-1:turn-2", 2, "Ada approved the budget.", {"document_id": "doc-1"})
        memory = Memory((source,))
        memory.apply(MemoryPatch("p1", upsert=(MemoryNode("root", "Budget decision", (source.id,)),)))
        self.assertEqual(memory.nodes["root"].source_citations, ((source.id, source.content_hash),))
        with self.assertRaisesRegex(ValueError, "source citation version"):
            memory.apply(MemoryPatch("bad-citation", expected_revision=1, upsert=(
                MemoryNode("child", "Approval detail", (source.id,),
                           source_citations=((source.id, "not-the-source-hash"),)),
            )))

    def test_concurrent_patches_cannot_both_commit_the_same_revision(self):
        from memorium import Memory, MemoryNode, MemoryPatch, SourceRecord

        source = SourceRecord("doc-1:turn-2", 2, "Ada approved the budget.", {"document_id": "doc-1"})
        memory = Memory((source,))
        patches = (
            MemoryPatch("p1", expected_revision=0, upsert=(MemoryNode("left", "Left", (source.id,)),)),
            MemoryPatch("p2", expected_revision=0, upsert=(MemoryNode("right", "Right", (source.id,)),)),
        )
        with ThreadPoolExecutor(max_workers=2) as pool:
            outcomes = [future.exception() for future in (pool.submit(memory.apply, patch) for patch in patches)]
        self.assertEqual(sum(error is None for error in outcomes), 1)
        self.assertEqual(memory.revision, 1)
        self.assertEqual(len(memory.nodes), 1)


if __name__ == "__main__":
    unittest.main()
