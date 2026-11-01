"""CL-5F expected-count arithmetic tests; never generate examples."""
from fractions import Fraction
import unittest

from integrations.continual_learning.cl5f_class_quota_draw_floor import (
    planned_task_draw_floors,
    uncapped_class_quota_draw_floor,
)


def fraction(value):
    return Fraction(value["numerator"], value["denominator"])


class ClassQuotaFloorTests(unittest.TestCase):
    def test_half_rate_lower_bound_is_requested_count(self):
        report = uncapped_class_quota_draw_floor(1, 2)
        self.assertEqual(fraction(report["lower_bound_expected_total_draws_for_task"]), 192)
        self.assertEqual(report["accepted_record_count"], 192)
        self.assertFalse(report["finite_attempt_cap_modelled"])

    def test_sparse_quota_floor_exceeds_accepted_records(self):
        report = uncapped_class_quota_draw_floor(1, 8)
        self.assertEqual(fraction(report["xor_class_probabilities"]["1"]), Fraction(7, 32))
        self.assertEqual(fraction(report["lower_bound_expected_total_draws_for_task"]), Fraction(3072, 7))
        self.assertGreater(fraction(report["lower_bound_expected_total_draws_for_task"]), 192)
        self.assertFalse(report["generator_success_probability_estimated"])

    def test_task_order_and_no_go(self):
        reports = planned_task_draw_floors()
        self.assertEqual(list(reports), ["T1", "T2", "T3", "T4"])
        self.assertTrue(all(r["decision"] == "NO_GO" for r in reports.values()))
        self.assertTrue(all(not r["dataset_generation_executed"] for r in reports.values()))

    def test_invalid_quota_and_probability_fail_closed(self):
        for args in ((0, 8), (1, 1), (True, 8), (1, False)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                uncapped_class_quota_draw_floor(*args)
        for params in ({"train_per_task": 1}, {"test_per_task": 0},
                       {"train_per_task": True}, {"test_per_task": 64.0}):
            with self.subTest(params=params), self.assertRaises(ValueError):
                uncapped_class_quota_draw_floor(1, 8, **params)


if __name__ == "__main__":
    unittest.main()
