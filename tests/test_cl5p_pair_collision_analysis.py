"""CL-5P mathematical and safety contracts; no generator imports."""
import math
import unittest

from integrations.continual_learning.cl5p_pair_collision_analysis import (
    analyze_existing, pair_collision, render_markdown,
)
from tests.test_cl5n_empirical_data_diagnostics import fixture


class PairCollisionTests(unittest.TestCase):
    def test_uniform_16bit_space_has_exact_pair_collision(self):
        expected = 1 / 65536
        result = pair_collision(0.5)
        self.assertAlmostEqual(result["raw_bernoulli"], expected)
        self.assertAlmostEqual(result["balanced_xor"], expected)
        self.assertAlmostEqual(result["effective_support"], 65536)

    def test_analytical_xor_conditioning(self):
        for p in (0.125, 0.25, 0.375, 0.5):
            q = 1 - p
            tail = (p*p + q*q) ** 14
            c0 = ((p**4 + q**4) / (p*p + q*q)**2) * tail
            c1 = tail / 2
            collision = (c0 + c1) / 4
            result = pair_collision(p)
            self.assertAlmostEqual(result["class0"], c0)
            self.assertAlmostEqual(result["class1"], c1)
            self.assertAlmostEqual(result["balanced_xor"], collision)
            self.assertAlmostEqual(result["raw_bernoulli"], (p*p + q*q)**16)

    def test_invalid_rates_are_rejected(self):
        for value in (0, 1, -1, math.nan, math.inf, True, None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                pair_collision(value)

    def test_existing_report_and_counters(self):
        rows = analyze_existing(fixture())
        self.assertEqual([r["duplicates"] for r in rows], [119, 42, 8, 2])
        self.assertEqual([r["eligible_proposals"] for r in rows],
                         [311, 234, 200, 194])
        summary = render_markdown(rows)
        self.assertIn("NO_GO", summary)
        self.assertIn("not** the expected rejection fraction", summary)
        self.assertIn("█", summary)

    def test_wrong_scope_and_corrupted_counts_fail_closed(self):
        data = fixture()
        data["model_training_executed"] = True
        with self.assertRaises(ValueError):
            analyze_existing(data)
        data = fixture()
        data["instrumented_counters"][0]["duplicate_rejections_after_class_check"] += 1
        with self.assertRaises(ValueError):
            analyze_existing(data)


if __name__ == "__main__":
    unittest.main()
