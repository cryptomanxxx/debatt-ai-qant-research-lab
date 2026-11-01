"""CL-7E static contract tests: never measure or train."""
from __future__ import annotations

from copy import deepcopy
import json
import unittest

from integrations.continual_learning.cl7e_measurement_boundary_audit import (
    PATH, CL6A, CL7A, CL7B, CL7C, CL7D, audit, validate_contract,
)


class CL7EMeasurementBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.values = [
            json.loads(path.read_text(encoding="utf-8"))
            for path in (PATH, CL6A, CL7A, CL7B, CL7C, CL7D)
        ]

    def reject(self, mutation):
        candidate = deepcopy(self.values[0])
        mutation(candidate)
        with self.assertRaises(ValueError):
            validate_contract(candidate, *self.values[1:])

    def test_current_document_stays_no_go(self):
        result = audit()
        self.assertEqual(result["decision"], "NO_GO")
        self.assertEqual(result["measurement_domains"], 5)
        self.assertEqual(result["open_parent_gates"], 9)
        self.assertEqual(result["open_scientific_findings"], 6)
        self.assertFalse(result["training_executed"])
        self.assertFalse(result["physical_energy_measured"])

    def test_unapproved_compute_and_extra_fields_fail(self):
        for field in ("training_authorized", "model_inference_authorized",
                      "dataset_generation_authorized", "hardware_energy_measured",
                      "new_results_present", "independent_scientific_signoff"):
            with self.subTest(field=field):
                self.reject(lambda d, field=field: d.__setitem__(field, True))
        self.reject(lambda d: d.__setitem__("experimental_result", 0.1))

    def test_no_proxy_can_be_claimed_as_measured_energy(self):
        self.reject(lambda d: d["counter_classes"][0].__setitem__(
            "status", "MEASURED"))
        self.reject(lambda d: d["counter_classes"][3].__setitem__(
            "status", "MEASURED"))
        self.reject(lambda d: d["counter_classes"][0].__setitem__(
            "claim", "Observed DRAM bytes and joules"))
        self.reject(lambda d: d["interpretation_fences"].__setitem__(
            "logical_operations_equated_to_joules", True))
        self.reject(lambda d: d["interpretation_fences"].__setitem__(
            "cpu_result_extrapolated_to_photonic", True))

    def test_no_unreviewed_timing_or_scoping_claim(self):
        self.reject(lambda d: d["future_observation_rules"].__setitem__(
            "wall_clock_repeat_count", 10))
        self.reject(lambda d: d["future_observation_rules"].__setitem__(
            "power_meter_and_idle_baseline", "CALIBRATED"))
        self.reject(lambda d: d["future_observation_rules"].__setitem__(
            "instrumented_vs_uninstrumented_control_required", False))
        self.reject(lambda d: d["future_observation_rules"].__setitem__(
            "missing_samples_policy", "DROP_FAILED_SEEDS"))


if __name__ == "__main__":
    unittest.main()
