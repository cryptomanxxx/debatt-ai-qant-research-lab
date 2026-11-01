"""CL-5Q static occupancy mathematics and provenance tests."""
import math
import unittest

from integrations.continual_learning.cl5p_pair_collision_analysis import pair_collision
from integrations.continual_learning.cl5q_iid_occupancy_reference import (
    analyze_report, grouped_masses, occupancy, render_markdown,
)
from tests.test_cl5n_empirical_data_diagnostics import fixture


class OccupancyTests(unittest.TestCase):
    def test_exact_distribution_mass_and_support(self):
        for p in (0.125, 0.25, 0.375, 0.5):
            groups = grouped_masses(p)
            self.assertEqual(sum(count for count, _ in groups), 2 ** 16)
            self.assertAlmostEqual(sum(count * mass for count, mass in groups), 1)
            self.assertAlmostEqual(
                sum(count * mass * mass for count, mass in groups),
                pair_collision(p)["balanced_xor"],
            )

    def test_first_and_second_proposals(self):
        for p in (0.125, 0.25, 0.375, 0.5):
            initial = occupancy(p, 0)
            self.assertAlmostEqual(initial["expected_distinct"], 0)
            self.assertAlmostEqual(initial["next_duplicate_probability"], 0)
            one = occupancy(p, 1)
            self.assertAlmostEqual(one["expected_distinct"], 1)
            self.assertAlmostEqual(
                one["next_duplicate_probability"],
                pair_collision(p)["balanced_xor"],
            )

    def test_uniform_occupancy_closed_form(self):
        n = 192
        result = occupancy(0.5, n)
        expected_distinct = 65536 * (1 - (1 - 1/65536) ** n)
        expected_next_dup = 1 - (1 - 1/65536) ** n
        self.assertAlmostEqual(result["expected_distinct"], expected_distinct, places=6)
        self.assertAlmostEqual(result["next_duplicate_probability"], expected_next_dup)

    def test_monotonicity_and_bounds(self):
        for p in (0.125, 0.25, 0.375, 0.5):
            points = [occupancy(p, n) for n in (0, 1, 16, 64, 192, 256)]
            self.assertEqual(
                [v["next_duplicate_probability"] for v in points],
                sorted(v["next_duplicate_probability"] for v in points),
            )
            for item in points:
                self.assertLessEqual(item["expected_distinct"], item["n"] + 1e-8)
                self.assertGreaterEqual(item["expected_repeats"], 0)

    def test_input_and_report_fail_closed(self):
        for invalid in (-1, 1.0, True, 100001):
            with self.subTest(n=invalid), self.assertRaises(ValueError):
                occupancy(.125, invalid)
        report = fixture()
        report["model_training_executed"] = True
        with self.assertRaises(ValueError):
            analyze_report(report)

    def test_no_generator_calls_and_disciplined_interpretation(self):
        rows = analyze_report(fixture())
        self.assertEqual(len(rows), 4)
        text = render_markdown(rows)
        self.assertIn("with-replacement", text)
        self.assertIn("cannot be used", text)
        self.assertIn("NO_GO", text)


if __name__ == "__main__":
    unittest.main()
