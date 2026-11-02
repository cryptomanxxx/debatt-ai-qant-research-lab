"""CL-19 proposal-only fixture mutation tests. No model imports/calls."""
import copy
import json
import unittest

from cl19_single_sample_forward_review import (
    SPEC, ExecutionNotAuthorized, attempt_model_execution,
    validate_review_only,
)


class TestCL19(unittest.TestCase):
    def setUp(self):
        self.spec = json.loads(SPEC.read_text(encoding="ascii"))

    def test_review_only_candidate_is_no_go(self):
        gate = validate_review_only(self.spec)
        self.assertEqual(gate["decision"], "NO_GO")
        self.assertEqual(gate["models_constructed"], 0)
        self.assertEqual(gate["forward_calls_executed"], 0)
        self.assertFalse(gate["inference_authorized"])
        self.assertFalse(gate["training_authorized"])

    def test_direct_execution_attempt_rejected(self):
        with self.assertRaises(ExecutionNotAuthorized):
            attempt_model_execution()

    def test_inference_authorization_change_rejected(self):
        changed = copy.deepcopy(self.spec)
        changed["inference_authorized"] = True
        with self.assertRaises(ValueError):
            validate_review_only(changed)

    def test_extra_forward_or_update_rejected(self):
        for field, value in (("forward_calls_max", 4), ("weight_updates_max", 1)):
            changed = copy.deepcopy(self.spec)
            changed["model_scope"][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_review_only(changed)

    def test_heldout_or_sample_substitution_rejected(self):
        for field, value in (("partition", "test"), ("zero_based_index", 1)):
            changed = copy.deepcopy(self.spec)
            changed["data_source"][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_review_only(changed)

    def test_fake_runtime_security_approval_rejected(self):
        changed = copy.deepcopy(self.spec)
        changed["future_runtime"]["host_boundary_security_approved"] = True
        with self.assertRaises(ValueError):
            validate_review_only(changed)


if __name__ == "__main__":
    unittest.main()
