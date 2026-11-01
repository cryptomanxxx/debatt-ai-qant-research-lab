"""CL-5C pure-arithmetic contract tests; no samples or training."""
import unittest

from integrations.continual_learning.cl5c_balanced_xor_bias import (
    balanced_xor_proposal_moments,
    planned_task_proposal_moments,
)


class BalancedXorBiasTests(unittest.TestCase):
    def test_half_rate_is_unchanged(self):
        moments = balanced_xor_proposal_moments(1, 2)
        self.assertEqual(moments["unconditioned_xor_one_probability"], {"numerator": 1, "denominator": 2})
        self.assertEqual(moments["unconditioned_mean_active_bits"], {"numerator": 8, "denominator": 1})
        self.assertEqual(moments["balanced_xor_mean_active_bits_before_uniqueness"], {"numerator": 8, "denominator": 1})
        self.assertFalse(moments["globally_unique_rejection_accounted_for"])

    def test_eighth_rate_shifts_under_class_balance(self):
        moments = balanced_xor_proposal_moments(1, 8)
        self.assertEqual(moments["unconditioned_xor_one_probability"], {"numerator": 7, "denominator": 32})
        self.assertEqual(moments["unconditioned_mean_active_bits"], {"numerator": 2, "denominator": 1})
        self.assertEqual(moments["balanced_xor_mean_active_bits_before_uniqueness"], {"numerator": 227, "denominator": 100})

    def test_reject_nonprobabilities_and_bool_inputs(self):
        for args in ((0, 8), (8, 8), (1, 0), (-1, 8), (True, 8), (1, False), (1.0, 8)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                balanced_xor_proposal_moments(*args)

    def test_planned_tasks_fail_closed_and_have_four_rows(self):
        moments = planned_task_proposal_moments()
        self.assertEqual(list(moments), ["T1", "T2", "T3", "T4"])
        self.assertTrue(all(row["decision"] == "NO_GO" for row in moments.values()))
        self.assertTrue(all(not row["dataset_generation_executed"] for row in moments.values()))


if __name__ == "__main__":
    unittest.main()
