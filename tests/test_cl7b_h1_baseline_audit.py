"""CL-7B no-compute preregistration tests; no model imports."""
from __future__ import annotations

from copy import deepcopy
import json
import unittest

from integrations.continual_learning.cl7b_h1_baseline_audit import (
    CL6A, CL7A, PATH, audit, validate_contract,
)


class CL7BBaselineAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract = json.loads(PATH.read_text(encoding="utf-8"))
        cls.parent = json.loads(CL6A.read_text(encoding="utf-8"))
        cls.critique = json.loads(CL7A.read_text(encoding="utf-8"))

    def reject(self, change):
        contract = deepcopy(self.contract)
        change(contract)
        with self.assertRaises(ValueError):
            validate_contract(contract, self.parent, self.critique)

    def test_current_document_is_no_go(self):
        report = audit()
        self.assertEqual(report["decision"], "NO_GO")
        self.assertEqual(report["baselines_specified"], 3)
        self.assertEqual(report["cl6a_open_gates"], 9)
        self.assertEqual(report["cl7a_open_findings"], 6)
        self.assertFalse(report["training_executed"])

    def test_execution_and_evidence_claims_fail(self):
        for key in ("training_authorized", "model_inference_authorized",
                    "dataset_generation_authorized", "new_results_present",
                    "independent_scientific_signoff"):
            with self.subTest(key=key):
                self.reject(lambda c, key=key: c.__setitem__(key, True))

    def test_heldout_selection_and_baseline_drift_fail(self):
        self.reject(lambda c: c["baseline_definitions"][1].__setitem__(
            "fit_partition", "heldout"))
        self.reject(lambda c: c["baseline_definitions"][1].__setitem__(
            "test_label_tuning", True))
        self.reject(lambda c: c["baseline_definitions"][2].__setitem__(
            "definition", "Observed accuracy 0.75 from model runs"))
        self.reject(lambda c: c["baseline_definitions"].pop())

    def test_quality_calibration_and_gates_fail_closed(self):
        self.reject(lambda c: c["quality_interpretation"].__setitem__(
            "quality_matched_cost_claim_permitted_now", True))
        self.reject(lambda c: c["quality_interpretation"].__setitem__(
            "paired_accuracy_difference_max_abs", 0.5))
        self.reject(lambda c: c["calibration"].__setitem__(
            "status", "COMPLETED"))
        self.reject(lambda c: c["gates"].__setitem__(
            "learnability_and_baselines", "CLOSED"))


if __name__ == "__main__":
    unittest.main()
