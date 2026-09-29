"""Deterministic offline tests for the selection measurement baseline."""
import unittest
from scripts.measure_researcher_selection import summarize, pareto, build_report


def result(candidate=444, control=448, params=6500, seeds=None):
    return {
        "experiment_id": "test",
        "configuration": {"seeds": seeds or [42, 43, 44, 45, 46], "epochs": 100},
        "summary": {
            "alpha5_w8": {"aggregate_qant_correct": candidate, "parameter_count": params},
            "alpha5_w16_control": {"aggregate_qant_correct": control, "parameter_count": 8550},
        },
    }


class MeasurementTests(unittest.TestCase):
    def test_paired_gate_boundary(self):
        self.assertTrue(summarize(result(candidate=438))[0]["passes_relative_gate_10"])
        self.assertFalse(summarize(result(candidate=437))[0]["passes_relative_gate_10"])

    def test_saving_and_margin(self):
        row = summarize(result())[0]
        self.assertEqual(row["correct_margin"], -4)
        self.assertEqual(row["parameter_saving"], 2050)

    def test_missing_or_invalid_data_fail_closed(self):
        self.assertIsNone(summarize({"summary": {}}))
        bad = result()
        bad["summary"]["alpha5_w8"]["aggregate_qant_correct"] = True
        self.assertIsNone(summarize(bad))

    def test_no_cross_seed_dominance(self):
        first = summarize(result(candidate=444, params=6500, seeds=[1, 2, 3, 4, 5]))[0]
        second = summarize(result(candidate=445, params=6400, seeds=[6, 7, 8, 9, 10]))[0]
        self.assertEqual(len(pareto([first, second])), 2)

    def test_same_seed_dominance(self):
        first = summarize(result(candidate=444, params=6500))[0]
        second = summarize(result(candidate=445, params=6400))[0]
        self.assertEqual(pareto([first, second]), [second])

    def test_empty_input(self):
        self.assertEqual(build_report([])["completed_candidate_observations"], 0)


if __name__ == "__main__":
    unittest.main()
