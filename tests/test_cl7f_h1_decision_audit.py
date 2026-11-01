"""CL-7F research decision matrix static tests (NO TRAINING)."""
from __future__ import annotations

from copy import deepcopy
import json
import unittest

from integrations.continual_learning.cl7f_h1_decision_audit import (
    PATH, CL6A, PREREQUISITES, audit, validate_matrix,
)


class CL7FDecisionMatrixTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.items = [
            json.loads(path.read_text(encoding="utf-8"))
            for path in (PATH, CL6A, *PREREQUISITES)
        ]

    def reject(self, change):
        candidate = deepcopy(self.items[0])
        change(candidate)
        with self.assertRaises(ValueError):
            validate_matrix(candidate, *self.items[1:])

    def test_matrix_preserves_no_go(self):
        result = audit()
        self.assertEqual(result["decision"], "NO_GO")
        self.assertEqual(result["open_cl6a_gates"], 9)
        self.assertEqual(result["open_cl7a_findings"], 6)
        self.assertEqual(result["experiment_run_count"], 0)
        self.assertFalse(result["model_training_executed"])

    def test_false_execution_or_review_signoff_fails(self):
        for field in ("execution_authorized", "training_authorized",
                      "model_inference_authorized", "dataset_generation_authorized",
                      "new_results_present", "hardware_energy_measured",
                      "independent_scientific_signoff"):
            with self.subTest(field=field):
                self.reject(lambda m, field=field: m.__setitem__(field, True))
        self.reject(lambda m: m.__setitem__("scientific_go_authorized", True))

    def test_gate_closure_missing_gate_and_false_evidence_fail(self):
        self.reject(lambda m: m["readiness_gates"][0].__setitem__(
            "status", "CLOSED"))
        self.reject(lambda m: m["readiness_gates"].pop())
        self.reject(lambda m: m["readiness_gates"][8].__setitem__(
            "next_evidence", "CI is green, training automatically approved"))
        self.reject(lambda m: m.__setitem__(
            "experiment_run_count_in_this_step", 1))

    def test_unsupported_claims_or_missing_prerequisites_fail(self):
        self.reject(lambda m: m["claims_not_supported"].remove(
            "measured_joules_per_learning_step"))
        self.reject(lambda m: m["prerequisites"].pop())
        self.reject(lambda m: m["proposed_sequence"].append(
            "Immediately run training without user authorization"))


if __name__ == "__main__":
    unittest.main()
