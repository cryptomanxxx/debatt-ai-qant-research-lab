"""CL-7D source-only causal contrast and no-GO guard tests."""
from __future__ import annotations

from copy import deepcopy
import json
import unittest

from integrations.continual_learning.cl7d_causal_pairing_audit import (
    PATH, CL6A, CL7A, CL7B, CL7C, audit, validate_contract,
)


class CL7DCausalPairingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract = json.loads(PATH.read_text(encoding="utf-8"))
        cls.parent = json.loads(CL6A.read_text(encoding="utf-8"))
        cls.critique = json.loads(CL7A.read_text(encoding="utf-8"))
        cls.baseline = json.loads(CL7B.read_text(encoding="utf-8"))
        cls.calibration = json.loads(CL7C.read_text(encoding="utf-8"))

    def reject(self, change):
        candidate = deepcopy(self.contract)
        change(candidate)
        with self.assertRaises(ValueError):
            validate_contract(candidate, self.parent, self.critique,
                              self.baseline, self.calibration)

    def test_no_results_or_training_are_claimed(self):
        report = audit()
        self.assertEqual(report["decision"], "NO_GO")
        self.assertEqual(report["causal_pairs_defined"], 2)
        self.assertEqual(report["parent_gates_open"], 9)
        self.assertEqual(report["open_scientific_findings"], 6)
        self.assertFalse(report["numerical_equivalence_verified"])
        self.assertFalse(report["training_executed"])

    def test_false_execution_and_unknown_fields_fail(self):
        for key in ("training_authorized", "model_inference_authorized",
                    "dataset_generation_authorized", "new_results_present",
                    "energy_measured", "independent_scientific_signoff"):
            with self.subTest(key=key):
                self.reject(lambda c, key=key: c.__setitem__(key, True))
        self.reject(lambda c: c.__setitem__("scientific_go_authorized", True))
        self.reject(lambda c: c.__setitem__("training_executed", True))

    def test_unsupported_causal_inference_fails(self):
        self.reject(lambda c: c["causal_pairs"][0].__setitem__(
            "left", "dense_sgd"))
        self.reject(lambda c: c["causal_pairs"][1].__setitem__(
            "status", "VERIFIED"))
        self.reject(lambda c: c["future_pairing_requirements"].__setitem__(
            "initial_state_identical", "VERIFIED"))
        self.reject(lambda c: c["future_pairing_requirements"].__setitem__(
            "numerical_tolerance", 1e-6))
        self.reject(lambda c: c["scientific_boundaries"].__setitem__(
            "quality_matched_cost_claim_permitted", True))
        self.reject(lambda c: c["failure_policy"].__setitem__(
            "divergent_outputs_or_states", "IGNORE"))

    def test_parent_protocol_drift_fails(self):
        altered = deepcopy(self.parent)
        altered["comparisons"][0]["left"] = "dense_sgd"
        with self.assertRaises(ValueError):
            validate_contract(self.contract, altered, self.critique,
                              self.baseline, self.calibration)


if __name__ == "__main__":
    unittest.main()
