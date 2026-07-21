import unittest

from scripts.aggregate_key_compensation_analysis import _transition_counts


class TransitionAggregationTest(unittest.TestCase):
    def test_paired_transitions_use_article_and_question_ids(self):
        top = [{
            "article_id": "a",
            "qa_results": {"results_per_question": [
                {"question_id": "q1", "is_correct": True, "model_choice": "A"},
                {"question_id": "q2", "is_correct": False, "model_choice": None},
            ]},
        }]
        random = [{
            "article_id": "a",
            "qa_results": {"results_per_question": [
                {"question_id": "q1", "is_correct": False, "model_choice": "B"},
                {"question_id": "q2", "is_correct": True, "model_choice": "C"},
            ]},
        }]
        counts = _transition_counts(top, random)
        self.assertEqual(counts["paired_questions"], 2)
        self.assertEqual(counts["correct_to_wrong"], 1)
        self.assertEqual(counts["wrong_to_correct"], 1)
        self.assertEqual(counts["parseable_to_parseable"], 1)
        self.assertEqual(counts["unanswered_to_parseable"], 1)

if __name__ == "__main__":
    unittest.main()
