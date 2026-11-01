"""CL-5E analytic probability tests only, with no sampled dataset."""
import unittest
from fractions import Fraction

from integrations.continual_learning.cl5e_conditional_collision_proxy import (
    conditional_collision_proxy,
    planned_collision_proxies,
)


def as_fraction(pair):
    return Fraction(pair["numerator"], pair["denominator"])


class ConditionalCollisionTests(unittest.TestCase):
    def test_uniform_half_rate(self):
        result = conditional_collision_proxy(1, 2)
        self.assertEqual(as_fraction(result["class_probabilities"]["0"]), Fraction(1, 2))
        self.assertEqual(as_fraction(result["class_probabilities"]["1"]), Fraction(1, 2))
        for value in result["same_class_independent_collision_probabilities"].values():
            self.assertEqual(as_fraction(value), Fraction(1, 32768))

    def test_sparse_proposal_is_more_concentrated_than_uniform(self):
        result = conditional_collision_proxy(1, 8)
        self.assertEqual(as_fraction(result["class_probabilities"]["1"]), Fraction(7, 32))
        for value in result["same_class_independent_collision_probabilities"].values():
            self.assertGreater(as_fraction(value), Fraction(1, 32768))
        self.assertFalse(result["global_uniqueness_or_rejection_attempts_modelled"])
        self.assertEqual(result["decision"], "NO_GO")

    def test_all_tasks_are_present(self):
        results = planned_collision_proxies()
        self.assertEqual(list(results), ["T1", "T2", "T3", "T4"])
        self.assertTrue(all(not x["dataset_generation_executed"] for x in results.values()))

    def test_reject_invalid_inputs(self):
        for args in ((0, 8), (8, 8), (-1, 8), (1, 0), (True, 8), (1, False)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                conditional_collision_proxy(*args)


if __name__ == "__main__":
    unittest.main()
