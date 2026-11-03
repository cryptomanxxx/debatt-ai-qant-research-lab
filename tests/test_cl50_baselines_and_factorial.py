"""Smoke test for the CL-50 experiment: one seed, one pass per task."""
import unittest

from experiments.cl50_baselines_and_factorial.run import ARMS, run


class CL50Smoke(unittest.TestCase):
    def test_one_seed_runs_and_sparse_arms_match_dense(self):
        results, _ = run(seeds=(101,), epoch_settings=(1,))
        block = results["settings"]["1"]
        self.assertTrue(all(block["bitwise_identity"].values()))
        self.assertEqual(set(block["arms"]), set(ARMS))
        for arm in block["arms"].values():
            self.assertTrue(0.0 <= arm["final_mean"]["mean"] <= 1.0)
        self.assertEqual(len(block["local_hybrid_diagnostics_after_each_task"]), 4)


if __name__ == "__main__":
    unittest.main()
