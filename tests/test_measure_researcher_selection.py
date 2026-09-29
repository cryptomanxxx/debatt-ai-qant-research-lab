"""Regression tests for Codex feedback on selection measurements."""
import json
import tempfile
import unittest
from pathlib import Path
from scripts.measure_researcher_selection import summarize, pareto, build_report


def result(candidate=444, control=448, params=6500, seeds=None):
    return {
        "experiment_id": "test",
        "dataset": {"name": "ECG200", "train_shape": [100, 96], "test_shape": [100, 96]},
        "configuration": {"seeds": seeds or [42, 43, 44, 45, 46], "epochs": 100,
                          "learning_rate": .001, "batch_size": 32, "frequencies": [1, 2]},
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

    def test_non_object_roots_are_skipped_without_aborting(self):
        with tempfile.TemporaryDirectory() as d:
            for name, data in [("a.json", None), ("b.json", []), ("c.json", 4),
                               ("d.json", result())]:
                (Path(d) / name).write_text(json.dumps(data))
            report = build_report(Path(d).glob("*.json"))
            self.assertEqual(report["completed_candidate_observations"], 1)
            self.assertEqual(len(report["skipped_files"]), 3)

    def test_malformed_candidate_rejects_entire_record(self):
        bad = result()
        bad["summary"]["alpha5_w12"] = {"parameter_count": 7000}
        self.assertIsNone(summarize(bad))
        bad = result()
        bad["summary"]["alpha5_w8"]["aggregate_qant_correct"] = True
        self.assertIsNone(summarize(bad))

    def test_no_cross_seed_dominance(self):
        first = summarize(result(candidate=444, params=6500, seeds=[1, 2, 3, 4, 5]))[0]
        second = summarize(result(candidate=445, params=6400, seeds=[6, 7, 8, 9, 10]))[0]
        self.assertEqual(len(pareto([first, second])), 2)

    def test_no_cross_protocol_dominance(self):
        first = summarize(result(candidate=444, params=6500))[0]
        different = result(candidate=445, params=6400)
        different["dataset"]["name"] = "OTHER"
        second = summarize(different)[0]
        self.assertEqual(len(pareto([first, second])), 2)
        different = result(candidate=445, params=6400)
        different["configuration"]["epochs"] = 50
        self.assertEqual(len(pareto([first, summarize(different)[0]])), 2)

    def test_same_protocol_dominance(self):
        first = summarize(result(candidate=444, params=6500))[0]
        second = summarize(result(candidate=445, params=6400))[0]
        self.assertEqual(pareto([first, second]), [second])

    def test_missing_protocol_fails_closed(self):
        bad = result()
        del bad["dataset"]
        self.assertIsNone(summarize(bad))

    def test_empty_input(self):
        self.assertEqual(build_report([])["completed_candidate_observations"], 0)


if __name__ == "__main__":
    unittest.main()
