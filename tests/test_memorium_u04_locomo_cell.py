import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class MemoriumU04LoCoMoCellTest(unittest.TestCase):
    def test_projects_public_tasks_and_source_turns_without_labels(self):
        root = Path(__file__).resolve().parents[1]
        script = root / "scripts" / "prepare_memorium_u04_locomo_cell.py"
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            writer = base / "writer.jsonl"
            questions = base / "questions.jsonl"
            source_manifest = base / "manifest.json"
            output = base / "out"
            writer.write_text(json.dumps({
                "sample_id": "conv-x", "speaker_a": "A", "speaker_b": "B",
                "sessions": [{"index": 1, "date_time": "today", "turns": [
                    {"dia_id": "D1:1", "speaker": "A", "text": "A fact.",
                     "blip_caption": "a photo"},
                ]}],
            }) + "\n", encoding="utf-8")
            questions.write_text(json.dumps({
                "id": "locomo:conv-x:0000", "sample_id": "conv-x", "question_index": 0,
                "official_qa": {"question": "What happened?", "answer": "A fact.",
                                "evidence": ["D1:1"], "category": 1},
            }) + "\n", encoding="utf-8")
            source_manifest.write_text(json.dumps({
                "status": "PASS_SOURCE_ONLY", "question_labels_opened": False,
            }), encoding="utf-8")
            # The production script enforces the frozen full-denominator contract.
            # Use the internal projection functions for this minimal unit contract.
            sys.path.insert(0, str(root))
            import scripts.prepare_memorium_u04_locomo_cell as cell
            writer_row = json.loads(writer.read_text())
            sources, _ = cell.source_rows([writer_row])
            tasks = cell.public_tasks([json.loads(questions.read_text())], 270727)
            self.assertEqual(sources[0]["id"], "locomo:conv-x:D1:1")
            self.assertEqual(sources[0]["locator"]["visual_representation"], "caption_only")
            self.assertEqual(tasks[0]["question"], "What happened?")
            self.assertEqual(tasks[0]["options"], [])
            self.assertNotIn("official_qa", tasks[0])
            self.assertNotIn("answer", json.dumps(tasks[0]).lower())

    def test_category_five_carries_the_official_two_way_option_pair(self):
        root = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(root))
        import scripts.prepare_memorium_u04_locomo_cell as cell
        from memorium.runtime.locomo import NO_INFO, choose_options

        row = {
            "id": "locomo:conv-x:0007", "sample_id": "conv-x", "question_index": 7,
            "official_qa": {"question": "When did Bo leave?", "adversarial_answer": "Bo never left.",
                            "evidence": [], "category": 5},
        }
        tasks = cell.public_tasks([row], 270727)

        # Free-form output cannot be scored against a two-way choice, so the pair is task input.
        self.assertEqual(tasks[0]["options"], choose_options(row, 270727))
        self.assertEqual(sorted(tasks[0]["options"]), sorted(["Bo never left.", NO_INFO]))
        self.assertNotIn("adversarial_answer", tasks[0])

    def test_category_five_query_matches_the_official_question_rendering(self):
        root = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(root))
        import scripts.prepare_memorium_u04_locomo_cell as cell
        from memorium import TaskSpec
        from memorium.runtime.locomo import locomo_question

        row = {
            "id": "locomo:conv-x:0007", "sample_id": "conv-x", "question_index": 7,
            "official_qa": {"question": "When did Bo leave?", "adversarial_answer": "Bo never left.",
                            "evidence": [], "category": 5},
        }
        task = cell.public_tasks([row], 270727)[0]

        self.assertEqual(TaskSpec(task["question"], tuple(task["options"])).public_query,
                         locomo_question(row, 270727))


if __name__ == "__main__":
    unittest.main()
