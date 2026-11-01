"""CL-3X only validates a draft, never trains or creates a dataset."""
import copy
import json
import unittest
from pathlib import Path

from integrations.continual_learning.cl2_paired_run_plan import load_candidate
from integrations.continual_learning.cl3x_h1_protocol_review import (
    PROTOCOL_PATH, READINESS_PATH, REQUIRED_GATES, review_summary, validate_draft,
)


class CL3XProtocolContracts(unittest.TestCase):
    def setUp(self):
        self.protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        self.readiness = json.loads(READINESS_PATH.read_text(encoding="utf-8"))
        self.candidate = load_candidate()

    def test_draft_remains_non_executable(self):
        result = review_summary()
        self.assertEqual("NO_GO", result["decision"])
        self.assertFalse(result["training_allowed"])
        self.assertFalse(result["runner_available"])
        self.assertFalse(result["physical_energy_claim_allowed"])
        self.assertEqual(5, result["seed_count"])
        self.assertEqual(4, result["task_count"])
        self.assertEqual(list(REQUIRED_GATES), result["missing_review_gates"])

    def test_refuse_hidden_compute_or_provenance_change(self):
        scenarios = (
            ("compute_approved", True),
            ("decision", "GO"),
            ("execution_entrypoint", "train.py"),
            ("results", {}),
        )
        for key, value in scenarios:
            changed = copy.deepcopy(self.protocol)
            changed[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_draft(changed, self.candidate, self.readiness)
        bad = copy.deepcopy(self.protocol)
        bad["study_design"]["dataset_source_sha256"] = "unreviewed"
        with self.assertRaises(ValueError):
            validate_draft(bad, self.candidate, self.readiness)
        bad = copy.deepcopy(self.protocol)
        bad["measurement_plan"]["meter_calibrated"] = True
        with self.assertRaises(ValueError):
            validate_draft(bad, self.candidate, self.readiness)

    def test_primary_contrast_does_not_conflate_learning_rule(self):
        primary = self.protocol["study_design"]["primary_comparison"]
        self.assertEqual("local_hybrid", primary["control"])
        self.assertEqual("sparse_local_hybrid", primary["candidate"])
        self.assertFalse(primary["implementation_equivalence_verified"])
        wrong = copy.deepcopy(self.protocol)
        wrong["study_design"]["primary_comparison"]["control"] = "dense_sgd"
        with self.assertRaises(ValueError):
            validate_draft(wrong, self.candidate, self.readiness)

    def test_current_readiness_is_still_locked(self):
        changed = copy.deepcopy(self.readiness)
        changed["decision"] = "GO"
        with self.assertRaises(ValueError):
            validate_draft(self.protocol, self.candidate, changed)


if __name__ == "__main__":
    unittest.main()
