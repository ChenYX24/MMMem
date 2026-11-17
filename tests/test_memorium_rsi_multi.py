import json
import random
import unittest

from memorium.training.rsi_multi import generate

LINES = ["[c0001] video (00:00:30-00:01:00): A man in a red coat opens a blue door.",
         "[c0002] video (00:01:00-00:01:30): He walks to a lake and sits on a bench."]
DIALOG = ["[D1:1] Ann (6 May 2023): I adopted a cat named Milo yesterday [= 5 May 2023].",
          "[D1:2] Bo (6 May 2023): Milo is a lovely name!"]


class Scripted:
    def __init__(self, gen, check):
        self.gen, self.check, self.prompts = gen, check, []

    def __call__(self, prompt, images, max_new):
        self.prompts.append((prompt, list(images)))
        return self.gen if max_new > 64 else self.check(prompt)


class ChoiceTest(unittest.TestCase):
    def test_choice_item_is_shuffled_and_verified(self):
        gen = json.dumps([{"type": "detail", "question": "What colour is the door?",
                           "options": ["blue", "red", "green", "white"], "answer_index": 0, "evidence": ["c0001"]},
                          {"type": "detail", "question": "Bad", "options": ["a", "b"], "answer_index": 0,
                           "evidence": ["c0001"]},
                          {"type": "detail", "question": "Where?", "options": ["lake", "sea", "hill", "park"],
                           "answer_index": 0, "evidence": ["c9999"]}])

        def check(prompt):  # an honest re-reader: finds the option that says blue
            for line in prompt.splitlines():
                if line.startswith("(") and "blue" in line:
                    return line[1]
            return "A"
        model = Scripted(gen, check)
        kept, tally = generate(LINES, ["f1.jpg"], "choice", "independent", model, n=3, rng=random.Random(0))
        self.assertEqual(len(kept), 1)
        self.assertEqual(tally["bad_type"], 1)
        self.assertEqual(tally["bad_evidence"], 1)
        item = kept[0]
        self.assertIn(f"({item['answer']}) blue", item["question"])
        self.assertEqual(model.prompts[0][1], ["f1.jpg"])  # generator saw the frames

    def test_choice_rejected_when_rereading_disagrees(self):
        gen = json.dumps([{"question": "What colour is the door?", "options": ["blue", "red", "green", "white"],
                           "answer_index": 0, "evidence": ["c0001"]}])
        kept, tally = generate(LINES, [], "choice", "independent",
                               Scripted(gen, lambda p: "Z"), n=1, rng=random.Random(0))
        self.assertEqual((kept, tally["inconsistent"]), ([], 1))


class SequenceTest(unittest.TestCase):
    def test_sequence_roles_and_families(self):
        gen = json.dumps([
            {"step": "fact", "question": "What is Ann's cat called?", "answer": "Milo", "evidence": ["D1:1"]},
            {"step": "paraphrase", "question": "Which name did Ann give her new cat?", "answer": "Milo",
             "evidence": ["D1:1"]},
            {"step": "other_need", "question": "When did Ann adopt the cat?", "answer": "5 May 2023",
             "evidence": ["D1:1"]},
            {"step": "near_miss", "question": "What is Bo's cat called?", "answer": "No information available",
             "distractor": "Milo", "evidence": []}])

        def check(prompt):
            q = prompt.rsplit("Question:", 1)[1]
            if "Bo's" in q:
                return "No information available"
            return "5 May 2023" if "When" in q else "Milo"
        kept, _ = generate(DIALOG, [], "free", "sequence", Scripted(gen, check), n=4, rng=random.Random(0),
                           unit="u", segment="s")
        self.assertEqual([k["type"] for k in kept], ["fact", "paraphrase", "other_need", "near_miss"])
        fams = [k["family"] for k in kept]
        self.assertEqual(fams[0], fams[1])
        self.assertEqual(len(set(fams)), 3)
        self.assertEqual(kept[3]["evidence"], [])
        self.assertEqual({k["sequence_id"] for k in kept}, {"u|s|seq"})  # round 1: the id RSI2_J was trained on
        again, _ = generate(DIALOG, [], "free", "sequence", Scripted(gen, check), n=4, rng=random.Random(0),
                            unit="u", segment="s", round=2)
        self.assertEqual({k["sequence_id"] for k in again}, {"u|s|seq|r2"})


if __name__ == "__main__":
    unittest.main()
