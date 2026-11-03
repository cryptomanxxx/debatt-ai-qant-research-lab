"""Smoke test for CL-52: one seed, one pass per task."""
import unittest

from experiments.cl52_cheap_winner_selection.run import ARMS, run


class CL52Smoke(unittest.TestCase):
    def test_one_seed_runs_and_exact_matches_cl51(self):
        results, _ = run(seeds=(101,), epochs=1)
        self.assertTrue(all(results["bitwise_identity"].values()))
        self.assertEqual(set(results["arms"]), set(ARMS))
        for arm in results["arms"].values():
            self.assertTrue(0.0 <= arm["final_mean"]["mean"] <= 1.0)
            self.assertTrue(0.0 <= arm["top_k_recall"] <= 1.0)


if __name__ == "__main__":
    unittest.main()
