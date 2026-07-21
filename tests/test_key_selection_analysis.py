import json
import math
import unittest

import torch

from compaction.analysis.key_selection import analyze_key_selections
from compaction.algorithms.key_selection_ablation import KeySelectionAblationCompaction


class KeySelectionMetricsTest(unittest.TestCase):
    def setUp(self):
        self.queries = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        self.top_keys = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        self.random_keys = torch.tensor([[1.0, 0.0], [-1.0, 0.0]])

    def test_overlap_geometry_routing_and_json_safety(self):
        result = analyze_key_selections(
            self.top_keys,
            self.random_keys,
            torch.tensor([0, 1]),
            torch.tensor([0, 2]),
            self.queries,
            torch.zeros(2),
            torch.zeros(2),
            candidate_count=4,
            dead_slot_threshold=0.2,
            spectrum_topk=2,
            hungarian_max_slots=8,
        )
        overlap = result["overlap"]
        self.assertEqual(overlap["intersection_size"], 1)
        self.assertAlmostEqual(overlap["overlap_ratio"], 0.5)
        self.assertAlmostEqual(overlap["jaccard"], 1 / 3)
        self.assertAlmostEqual(overlap["expected_overlap_ratio"], 0.5)
        self.assertAlmostEqual(overlap["expected_intersection_size"], 1.0)
        self.assertAlmostEqual(overlap["normalized_overlap"], 1.0)
        self.assertAlmostEqual(result["routing"]["raw"]["routing_gram_error"],
                               result["routing"]["fitted_bias"]["routing_gram_error"])
        json.dumps(result, allow_nan=False)

    def test_degenerate_routing_has_json_safe_condition_number(self):
        repeated = torch.ones(3, 2)
        result = analyze_key_selections(
            repeated, repeated.clone(), torch.arange(3), torch.arange(3),
            torch.zeros(4, 2), torch.zeros(3), torch.zeros(3), candidate_count=3,
        )
        spectrum = result["routing"]["raw"]["top"]["spectrum"]
        self.assertTrue(spectrum["rank_deficient"])
        self.assertIsNone(spectrum["condition_number"])
        self.assertAlmostEqual(spectrum["stable_rank"], 1.0, places=5)
        json.dumps(result, allow_nan=False)


class KeySelectionAblationTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.K = torch.randn(8, 4)
        self.V = torch.randn(8, 4)
        self.Q = torch.randn(12, 4)
        self.common = dict(
            selection_method="random",
            nnls_iters=0,
            nnls_lower_bound=math.exp(-3),
            nnls_upper_bound=math.exp(3),
            key_seed=41,
        )

    def _run(self, use_fitted_bias, value_method, analysis=False, key_seed=41):
        algorithm = KeySelectionAblationCompaction(
            **{**self.common, "key_seed": key_seed},
            use_fitted_bias=use_fitted_bias,
            value_method=value_method,
            enable_key_selection_analysis=analysis,
            hungarian_max_slots=8,
        )
        algorithm.set_selection_context(2, 3, 1)
        result = algorithm.compute_compacted_cache(self.K, self.V, self.Q, 3)
        return algorithm, result

    def test_direct_value_and_support_are_exact_and_reproducible(self):
        _, (_, beta_a, value_a, indices_a) = self._run(False, "direct")
        _, (_, beta_b, value_b, indices_b) = self._run(True, "direct")
        self.assertEqual(indices_a, indices_b)
        self.assertTrue(torch.equal(value_a, self.V[indices_a]))
        self.assertTrue(torch.equal(value_b, self.V[indices_b]))
        self.assertTrue(torch.equal(beta_a, torch.zeros_like(beta_a)))
        self.assertFalse(torch.equal(beta_b, torch.zeros_like(beta_b)))

    def test_zero_bias_value_is_refit_and_seed_is_independent(self):
        _, (_, beta_zero, value_zero, indices_zero) = self._run(False, "fitted")
        _, (_, beta_fit, value_fit, indices_fit) = self._run(True, "fitted")
        self.assertEqual(indices_zero, indices_fit)
        self.assertTrue(torch.equal(beta_zero, torch.zeros_like(beta_zero)))
        self.assertFalse(torch.allclose(value_zero, self.V[indices_zero]))
        self.assertFalse(torch.allclose(value_zero, value_fit))
        _, (_, _, _, other_indices) = self._run(False, "direct", key_seed=42)
        self.assertNotEqual(indices_zero, other_indices)

    def test_full_analysis_contains_all_eight_combinations(self):
        algorithm = KeySelectionAblationCompaction(
            selection_method="top", use_fitted_bias=True, value_method="fitted",
            nnls_iters=0, key_seed=41, enable_key_selection_analysis=True,
            hungarian_max_slots=8,
        )
        algorithm.set_selection_context(0, 0, 0)
        algorithm.compute_compacted_cache(self.K, self.V, self.Q, 3)
        analysis = algorithm.last_key_selection_analysis
        self.assertEqual(set(analysis["combination_train_metrics"]), {"top", "random"})
        for selection in ("top", "random"):
            self.assertEqual(len(analysis["combination_train_metrics"][selection]), 4)
        json.dumps(analysis, allow_nan=False)

    def test_algorithm_does_not_consume_global_torch_rng(self):
        torch.manual_seed(1234)
        expected = torch.rand(5)
        torch.manual_seed(1234)
        self._run(True, "fitted")
        actual = torch.rand(5)
        self.assertTrue(torch.equal(expected, actual))


if __name__ == "__main__":
    unittest.main()
