"""Artificial arithmetic-only fixtures; NEVER experimental model results."""
import unittest

from integrations.continual_learning.cl3d_analysis_contract import (
    SEEDS,
    paired_summary,
    summarize_accuracy,
)


def fixture(final: float) -> list[list[float]]:
    # Synthetic numbers solely to verify calculations (NO inference).
    matrix = [[0.5] * 4 for _ in range(5)]
    for checkpoint in range(1, 5):
        for task in range(checkpoint):
            matrix[checkpoint][task] = final
    return matrix


class CL3DAnalysisContracts(unittest.TestCase):
    def test_forgetting_excludes_last_task(self):
        matrix = fixture(0.75)
        matrix[2][0] = 0.9
        result = summarize_accuracy(matrix)
        self.assertAlmostEqual(0.75, result["final_mean_accuracy"])
        self.assertAlmostEqual(0.15, result["forgetting_by_task"][0])
        self.assertEqual(0.0, result["forgetting_by_task"][1])
        self.assertIsNone(result["forgetting_by_task"][3])

    def test_paired_gates_and_descriptive_only(self):
        dense = {seed: fixture(0.70) for seed in SEEDS}
        local = {seed: fixture(0.68) for seed in SEEDS}
        result = paired_summary(dense, local)
        self.assertTrue(result["quality_gate_all_seeds"])
        self.assertTrue(result["parity_gate_all_seeds"])
        self.assertTrue(result["logical_cost_comparison_interpretable"])
        self.assertAlmostEqual(-0.02, result["mean_local_minus_dense_accuracy"])
        self.assertAlmostEqual(0.0, result["paired_sample_sd"])
        self.assertFalse(result["physical_energy_inference_allowed"])
        self.assertFalse(result["statistical_significance_inference_allowed"])
        local[101] = fixture(0.55)
        result = paired_summary(dense, local)
        self.assertFalse(result["quality_gate_all_seeds"])
        self.assertFalse(result["parity_gate_all_seeds"])
        self.assertFalse(result["logical_cost_comparison_interpretable"])

    def test_rejects_missing_or_invalid_data(self):
        dense = {seed: fixture(0.70) for seed in SEEDS}
        with self.assertRaises(ValueError):
            paired_summary(dense, {101: fixture(0.7)})
        with self.assertRaises(ValueError):
            summarize_accuracy([[1.0] * 4] * 4)
        with self.assertRaises(ValueError):
            summarize_accuracy([[float("nan")] * 4] * 5)


if __name__ == "__main__":
    unittest.main()
