"""Discrete theoretical crossover checks only, no training or hardware."""
import copy
import unittest

from integrations.continual_learning.cl3r_visualization import build_preview
from integrations.continual_learning.cl3t_crossover_analysis import analyze_crossover


class CL3TCrossoverContracts(unittest.TestCase):
    def test_sparse_inference_crossover_precedes_training_crossover(self):
        summary = analyze_crossover(build_preview())
        self.assertTrue(summary["no_training_or_inference_performed"])
        self.assertFalse(summary["accuracy_tradeoff_verified"])
        keyed = {(r["phase"], r["arm"]): r for r in summary["comparisons"]}
        self.assertEqual(14, keyed[("inference", "sparse_local_hybrid")]["first_not_better_activity"])
        self.assertEqual(13, keyed[("inference", "sparse_shared_index")]["last_strictly_better_activity"])
        self.assertEqual(16, keyed[("training", "sparse_local_hybrid")]["first_not_better_activity"])
        self.assertEqual(16, keyed[("training", "sparse_shared_index")]["first_not_better_activity"])
        self.assertEqual(0, keyed[("inference", "local_hybrid")]["first_not_better_activity"])

    def test_missing_sweep_or_nonmonotone_advantage_rejected(self):
        original = build_preview()
        missing = copy.deepcopy(original)
        missing["series"].pop()
        with self.assertRaises(ValueError):
            analyze_crossover(missing)
        inconsistent = copy.deepcopy(original)
        key = "training_logical_payload_bytes_per_assumed_update"
        inconsistent["series"][1]["by_arm"]["sparse_shared_index"][key] = (
            inconsistent["series"][1]["by_arm"]["dense_sgd"][key] + 1
        )
        with self.assertRaises(ValueError):
            analyze_crossover(inconsistent)


if __name__ == "__main__":
    unittest.main()
