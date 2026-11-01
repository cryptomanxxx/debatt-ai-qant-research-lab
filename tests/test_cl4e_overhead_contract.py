"""CL-4E synthetic timing analysis contracts; no calls to models or clocks."""
import copy
import unittest

from integrations.continual_learning.cl4e_overhead_contract import (
    SCOPE, PATTERN, analyze_synthetic_pairs,
)


def fixture():
    durations = ((100, 115, 117, 102), (105, 119, 121, 107))
    return [{
        "round": n,
        "origin": "SYNTHETIC_FIXTURE",
        "scope": SCOPE,
        "samples": [{"mode": mode, "elapsed_ns": elapsed}
                    for mode, elapsed in zip(PATTERN, values)],
    } for n, values in enumerate(durations)]


class CL4EOverheadContracts(unittest.TestCase):
    def test_paired_estimator_uses_each_control_and_traced_sample(self):
        result = analyze_synthetic_pairs(fixture())
        self.assertEqual(2, result["round_count"])
        self.assertEqual(15.5, result["median_overhead_ns"])
        self.assertGreater(result["median_ratio"], 1)
        self.assertFalse(result["instrumentation_overhead_measured"])
        self.assertFalse(result["observed_model_runtime"])
        self.assertFalse(result["training_allowed"])
        self.assertFalse(result["energy_measured"])

    def test_rejects_missing_ambiguous_or_nonpositive_durations(self):
        for mutation in (
            lambda x: x[0]["samples"][0].update(elapsed_ns=0),
            lambda x: x[0]["samples"][1].update(mode="plain"),
            lambda x: x[0]["samples"][2].update(elapsed_ns=1.5),
            lambda x: x[1].update(round=5),
            lambda x: x[1].update(origin="MEASURED_GITHUB_RUNNER"),
            lambda x: x[0].update(scope="hardware_energy"),
        ):
            rows = fixture()
            mutation(rows)
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                analyze_synthetic_pairs(rows)
        with self.assertRaises(ValueError):
            analyze_synthetic_pairs(fixture()[:1])
        with self.assertRaises(ValueError):
            analyze_synthetic_pairs(fixture() * 11)

    def test_negative_overhead_not_silently_clamped_to_zero(self):
        rows = fixture()
        for row in rows:
            for sample in row["samples"]:
                sample["elapsed_ns"] = 100 if sample["mode"] == "plain" else 90
        result = analyze_synthetic_pairs(rows)
        self.assertEqual(-10, result["median_overhead_ns"])


if __name__ == "__main__":
    unittest.main()
