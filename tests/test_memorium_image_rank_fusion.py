from types import SimpleNamespace
import unittest
from memorium import Memory, MemoryNode, MemoryPatch, SourceRecord, TaskSpec
from memorium.core import View
from memorium.u05 import ArmConfig, select, run_task
from memorium.unit import BuiltUnit, UnitConfig
from memorium.views import ViewLedger


class ImageRankFusionTest(unittest.TestCase):
    def setUp(self):
        sources = [SourceRecord(str(i), i, 'flower', {'sample_id':'u'}) for i in range(3)]
        memory = Memory(sources)
        nodes = [MemoryNode(str(i), 'u', source_refs=(str(i),),
                            views=(View('T0', 'flower '+str(i), (str(i),), 'source', 1),),
                            canvas={'type': 'segment'}) for i in range(3)]
        memory.apply(MemoryPatch('build', tuple(nodes)))
        self.unit = BuiltUnit('u', memory, ['0', '1', '2'], [], {}, ViewLedger(), UnitConfig())
        self.arm = ArmConfig(token_budget=1, write_back=False, hops=0)
        self.reranker = lambda q, docs: [3-int(d.rsplit(' ',1)[-1]) for d in docs]

    def test_picture_rank_survives_text_reranking(self):
        old = select(self.unit, 'which flower?', ['0', '1', '2'], self.arm, self.reranker)
        new = select(self.unit, 'which flower?', ['0', '1', '2'], self.arm, self.reranker, ['2'])
        self.assertEqual(old.nodes, ['0'])
        self.assertEqual(new.nodes, ['2'])

    def test_no_image_keeps_previous_selection(self):
        self.assertEqual(select(self.unit, 'q', ['0', '1', '2'], self.arm, self.reranker).nodes, ['0'])

    def test_unknown_image_nodes_cannot_add_evidence(self):
        result = select(self.unit, 'q', ['0', '1', '2'], self.arm, self.reranker, ['absent', '2', '2'])
        self.assertEqual(result.nodes, ['2'])
        self.assertEqual(result.text_tokens, 1)

    def test_flag_routes_picture_rank_through_real_task_loop(self):
        index = SimpleNamespace(rank=lambda *a, **k: ['0', '1', '2'])
        visual = SimpleNamespace(rank=lambda *a, **k: [], rank_item=lambda *a, **k: ['2'])
        reader = lambda *a: {'output': 'Answer: flower\nEvidence: 2'}
        old = run_task(self.unit, TaskSpec('flower'), 'a', self.arm,
                       text_index=index, visual_index=visual, query_image='picture',
                       reader=reader, reranker=self.reranker)
        new = run_task(self.unit, TaskSpec('flower'), 'b',
                       ArmConfig(token_budget=1, write_back=False, hops=0, preserve_query_image_rank=True),
                       text_index=index, visual_index=visual, query_image='picture',
                       reader=reader, reranker=self.reranker)
        self.assertEqual(old.selected, ['0'])
        self.assertEqual(new.selected, ['2'])


if __name__ == '__main__':
    unittest.main()
