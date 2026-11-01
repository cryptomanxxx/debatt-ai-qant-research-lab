"""CL-5O analytical contracts; entirely static, no data generation."""
import math
import unittest

from integrations.continual_learning.cl5o_class_conditioning_analysis import (
    compare_existing_report, render_markdown, xor_balanced_activity,
)
from tests.test_cl5n_empirical_data_diagnostics import fixture


class ClassConditioningTests(unittest.TestCase):
    def test_half_probability_is_symmetric(self):
        self.assertAlmostEqual(xor_balanced_activity(0.5), 0.5)

    def test_exact_pair_formula_independent_of_other_bits(self):
        for p in (0.125, 0.25, 0.375, 0.5):
            q = 1 - p
            expected = (14 * p + (2 * p * p / (p * p + q * q) + 1) / 2) / 16
            self.assertAlmostEqual(xor_balanced_activity(p), expected)
        self.assertGreater(xor_balanced_activity(0.125), 0.125)

    def test_reject_invalid_probabilities(self):
        for bad in (0, 1, -1, math.inf, math.nan, True, None):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                xor_balanced_activity(bad)

    def test_existing_observations_unchanged_and_not_overattributed(self):
        rows = compare_existing_report(fixture())
        self.assertEqual([r["task"] for r in rows], ["T1", "T2", "T3", "T4"])
        self.assertAlmostEqual(rows[0]["observed"], 555 / 3072)
        self.assertAlmostEqual(rows[3]["class_balanced"], 0.5)
        rendered = render_markdown(rows)
        self.assertIn("unexplained residual", rendered.lower())
        self.assertIn("must **not**", rendered)
        self.assertIn("NO_GO", rendered)

    def test_rejects_invalid_source_report(self):
        report = fixture()
        report["model_training_executed"] = True
        with self.assertRaises(ValueError):
            compare_existing_report(report)


if __name__ == "__main__":
    unittest.main()
