"""Smoke test for CL-51: one seed, one pass per task."""
import unittest

from experiments.cl51_local_rules_and_hidden_sparsity.run import PART_A, PART_B, run


class CL51Smoke(unittest.TestCase):
    def test_one_seed_runs_and_reproduces_cl2_classes(self):
        results, _ = run(seeds=(101,), epochs=1)
        self.assertTrue(all(results["bitwise_identity_with_cl2_classes"].values()))
        self.assertEqual(set(results["arms"]), set(PART_A) | set(PART_B))
        for arm in results["arms"].values():
            self.assertTrue(0.0 <= arm["final_mean"]["mean"] <= 1.0)


if __name__ == "__main__":
    unittest.main()
