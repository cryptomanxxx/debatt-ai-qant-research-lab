"""CL-28 no-model scientific acceptance fixture tests."""
import copy
import json
import math
import unittest

from cl28_forward_only_criteria import (
    SPEC, ARMS, audit_synthetic_fixture, validate_review_only,
)


class TestCL28(unittest.TestCase):
    def setUp(self):
        self.spec = json.loads(SPEC.read_text(encoding="ascii"))
        self.record = {
            "synthetic_fixture": True,
            "traces": {
                arm: {
                    "arm": arm, "initial_weights": [0.0] * 154,
                    "hidden": [0.0] * 8, "logits": [0.0] * 2,
                    "active_indices": [2, 8],
                }
                for arm in ARMS
            },
        }

    def test_no_go_without_model_imports(self):
        self.assertEqual(validate_review_only(self.spec)["decision"], "NO_GO")
        result = audit_synthetic_fixture(self.spec, self.record)
        self.assertTrue(result["candidate_parity_within_tolerance"])
        self.assertEqual(result["compared_values"], 40)
        self.assertFalse(result["observations_authenticated"])

    def test_tiny_roundoff_accepted_in_fixture_only(self):
        changed = copy.deepcopy(self.record)
        changed["traces"][ARMS[1]]["logits"][0] = 5e-13
        self.assertTrue(audit_synthetic_fixture(
            self.spec, changed)["candidate_parity_within_tolerance"])

    def test_numerical_mismatch_rejected_by_candidate(self):
        changed = copy.deepcopy(self.record)
        changed["traces"][ARMS[1]]["logits"][0] = 1e-6
        result = audit_synthetic_fixture(self.spec, changed)
        self.assertFalse(result["candidate_parity_within_tolerance"])
        self.assertIn("logits[0]", result["first_mismatch"])

    def test_unmatched_initial_weights_rejected(self):
        changed = copy.deepcopy(self.record)
        changed["traces"][ARMS[2]]["initial_weights"][0] = 0.1
        with self.assertRaisesRegex(ValueError, "unequal initialization"):
            audit_synthetic_fixture(self.spec, changed)

    def test_nonfinite_logits_rejected(self):
        changed = copy.deepcopy(self.record)
        changed["traces"][ARMS[0]]["logits"][0] = math.nan
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            audit_synthetic_fixture(self.spec, changed)

    def test_different_active_indices_rejected(self):
        changed = copy.deepcopy(self.record)
        changed["traces"][ARMS[1]]["active_indices"] = [2]
        with self.assertRaisesRegex(ValueError, "active index"):
            audit_synthetic_fixture(self.spec, changed)

    def test_loss_or_optimizer_field_rejected(self):
        changed = copy.deepcopy(self.record)
        changed["traces"][ARMS[0]]["loss"] = 0.1
        with self.assertRaisesRegex(ValueError, "trace shape"):
            audit_synthetic_fixture(self.spec, changed)

    def test_nonfixture_claim_rejected(self):
        changed = copy.deepcopy(self.record)
        changed["synthetic_fixture"] = False
        with self.assertRaisesRegex(ValueError, "fixture provenance"):
            audit_synthetic_fixture(self.spec, changed)

    def test_false_scientific_approval_rejected(self):
        changed = copy.deepcopy(self.spec)
        changed["tolerance"]["independently_approved"] = True
        with self.assertRaisesRegex(ValueError, "tolerance"):
            validate_review_only(changed)

    def test_nonzero_training_budget_rejected(self):
        changed = copy.deepcopy(self.spec)
        changed["execution"]["weight_updates_max"] = 1
        with self.assertRaisesRegex(ValueError, "execution budget"):
            validate_review_only(changed)


if __name__ == "__main__":
    unittest.main()
