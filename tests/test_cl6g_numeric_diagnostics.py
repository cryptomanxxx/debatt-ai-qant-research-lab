"""CL-6G literal scalar-only fixtures: no model import, inference or training."""
from __future__ import annotations

import unittest

from integrations.continual_learning.cl6g_numeric_diagnostics import diagnose_pair


class CL6GNumericDiagnosticsTests(unittest.TestCase):
    def test_identical_finite_scalars_are_descriptive_not_approved(self):
        report = diagnose_pair([0.0, 1.25, -2.0], [0.0, 1.25, -2.0])
        self.assertEqual(report["count"], 3)
        self.assertEqual(report["exact_equal_count"], 3)
        self.assertEqual(report["max_absolute_difference"], 0.0)
        self.assertEqual(report["max_symmetric_relative_difference"], 0.0)
        self.assertEqual(report["scientific_decision"], "NO_GO")
        self.assertFalse(report["tolerance_approved"])
        self.assertFalse(report["equivalence_established"])
        self.assertFalse(report["model_executed"])

    def test_signed_and_zero_difference_metrics(self):
        report = diagnose_pair((0.0, 2.0, -2.0), (1.0, 1.0, -1.0))
        self.assertEqual(report["max_absolute_difference"], 1.0)
        self.assertEqual(report["max_symmetric_relative_difference"], 1.0)
        self.assertEqual(report["exact_equal_count"], 0)

    def test_invalid_shapes_and_types_fail_closed(self):
        inputs = [
            ([], []),
            ([1.0], []),
            ([1.0], [1.0, 2.0]),
            ([True], [1.0]),
            ([1.0], ["1.0"]),
            ([[1.0]], [[1.0]]),
            ((x for x in [1.0]), [1.0]),
        ]
        for left, right in inputs:
            with self.subTest(left=repr(left), right=repr(right)):
                with self.assertRaises(ValueError):
                    diagnose_pair(left, right)

    def test_nonfinite_and_overflow_differences_fail_closed(self):
        for a, b in [
            (float("nan"), 1.0),
            (1.0, float("inf")),
            (float("-inf"), 0.0),
            (1e308, -1e308),
            (10 ** 1000, 1.0),
        ]:
            with self.subTest(a=str(a), b=str(b)):
                with self.assertRaises(ValueError):
                    diagnose_pair([a], [b])


if __name__ == "__main__":
    unittest.main()
