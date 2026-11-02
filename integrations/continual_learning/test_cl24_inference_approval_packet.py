"""CL-24 negative safety tests; no model imports or numerical execution."""
import copy
import json
import sys
import unittest

from cl24_inference_approval_packet import (
    SPEC, InferenceNotAuthorized, attempt_inference, audit_current_files,
    validate_review_only,
)


class TestCL24(unittest.TestCase):
    def setUp(self):
        self.packet = json.loads(SPEC.read_text(encoding="ascii"))

    def test_exact_no_go_packet(self):
        result = audit_current_files()
        self.assertEqual(result["decision"], "NO_GO")
        self.assertEqual(result["forward_calls_executed"], 0)
        self.assertEqual(result["models_imported"], 0)
        self.assertEqual(result["source_files_verified"], 10)

    def test_attempt_inference_always_refused(self):
        with self.assertRaises(InferenceNotAuthorized):
            attempt_inference()
        with self.assertRaises(InferenceNotAuthorized):
            attempt_inference(object(), authorization="pretend")
        self.assertFalse(any(name in sys.modules for name in (
            "integrations.continual_learning.cl2_local_mlp",
            "integrations.continual_learning.cl2_mlp_baseline",
        )))

    def test_fake_authorization_rejected(self):
        changed = copy.deepcopy(self.packet)
        changed["inference_authorized"] = True
        with self.assertRaisesRegex(ValueError, "approval"):
            validate_review_only(changed)

    def test_increased_forward_budget_rejected(self):
        changed = copy.deepcopy(self.packet)
        changed["models"]["forward_calls_max"] = 4
        with self.assertRaisesRegex(ValueError, "scientific scope"):
            validate_review_only(changed)

    def test_training_steps_rejected(self):
        changed = copy.deepcopy(self.packet)
        changed["models"]["training_steps_max"] = 1
        with self.assertRaisesRegex(ValueError, "scientific scope"):
            validate_review_only(changed)

    def test_fake_final_image_digest_rejected(self):
        changed = copy.deepcopy(self.packet)
        changed["runtime"]["final_model_image_digest"] = "sha256:" + "0" * 64
        with self.assertRaisesRegex(ValueError, "runtime"):
            validate_review_only(changed)

    def test_fake_signoff_rejected(self):
        changed = copy.deepcopy(self.packet)
        changed["approvals"]["security_signoff"] = True
        with self.assertRaisesRegex(ValueError, "signoff"):
            validate_review_only(changed)

    def test_fake_observation_rejected(self):
        changed = copy.deepcopy(self.packet)
        changed["outputs"]["observations_recorded"] = 1
        with self.assertRaisesRegex(ValueError, "output"):
            validate_review_only(changed)


if __name__ == "__main__":
    unittest.main()
