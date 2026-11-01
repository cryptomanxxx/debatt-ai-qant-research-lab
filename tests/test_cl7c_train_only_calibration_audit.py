"""CL-7C no-compute calibration contract tests."""
from __future__ import annotations

from copy import deepcopy
import json
import unittest

from integrations.continual_learning.cl7c_train_only_calibration_audit import (
    CL6A, CL7A, CL7B, PATH, audit, validate_protocol,
)


class CL7CCalibrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = json.loads(PATH.read_text(encoding="utf-8"))
        cls.parent = json.loads(CL6A.read_text(encoding="utf-8"))
        cls.critique = json.loads(CL7A.read_text(encoding="utf-8"))
        cls.baseline = json.loads(CL7B.read_text(encoding="utf-8"))

    def rejected(self, change):
        candidate = deepcopy(self.protocol)
        change(candidate)
        with self.assertRaises(ValueError):
            validate_protocol(candidate, self.parent, self.critique, self.baseline)

    def test_current_protocol_is_not_executable(self):
        report = audit()
        self.assertEqual(report["decision"], "NO_GO")
        self.assertEqual(report["open_parent_gates"], 9)
        self.assertEqual(report["open_scientific_findings"], 6)
        self.assertEqual(report["calibration_status"], "NOT_RUN")
        self.assertFalse(report["training_executed"])

    def test_execution_and_results_are_forbidden(self):
        for key in ("model_construction_authorized", "model_inference_authorized",
                    "training_authorized", "dataset_generation_authorized",
                    "new_results_present", "independent_scientific_signoff"):
            with self.subTest(key=key):
                self.rejected(lambda d, key=key: d.__setitem__(key, True))

    def test_no_test_label_or_checkpoint_leakage(self):
        self.rejected(lambda d: d["calibration_split"].__setitem__(
            "heldout_test_labels_eligible", True))
        self.rejected(lambda d: d["selection_fences"].__setitem__(
            "heldout_checkpoint_selection", True))
        self.rejected(lambda d: d["selection_fences"].__setitem__(
            "cross_seed_best_run_selection", True))
        self.rejected(lambda d: d["calibration_split"].__setitem__(
            "split_unit", "individual_row"))

    def test_no_false_split_or_calibration_readiness(self):
        self.rejected(lambda d: d.__setitem__("calibration_status", "COMPLETE"))
        self.rejected(lambda d: d["calibration_split"].__setitem__(
            "per_seed_split_indices_frozen", True))
        self.rejected(lambda d: d["acceptance_rules"].__setitem__(
            "train_only_learnability_threshold", 0.65))
        self.rejected(lambda d: d["acceptance_rules"].__setitem__(
            "quality_matched_cost_claim_permitted", True))
        self.rejected(lambda d: d["future_control_definitions"][1].__setitem__(
            "executed", True))

    def test_unknown_execution_and_readiness_keys_are_rejected(self):
        for key in ("training_executed", "scientific_go_authorized",
                    "data_access_authorized", "hardware_energy_joules"):
            with self.subTest(key=key):
                self.rejected(lambda d, key=key: d.__setitem__(key, True))
        self.rejected(lambda d: d.__setitem__("miscellaneous", "harmless"))
        self.rejected(lambda d: d.pop("calibration_status"))

    def test_parent_go_mutations_are_rejected(self):
        altered_parent = deepcopy(self.parent)
        altered_parent["decision"] = "GO"
        with self.assertRaises(ValueError):
            validate_protocol(self.protocol, altered_parent, self.critique, self.baseline)


if __name__ == "__main__":
    unittest.main()
