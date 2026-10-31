"""CL-2 safety and metric unit tests. No data creation or model training."""
import copy
import unittest

from integrations.continual_learning.cl2_protocol import (
    forgetting_by_task,
    load_and_validate,
    validate_draft,
)


class CL2ProtocolContractTests(unittest.TestCase):
    def test_draft_is_valid_but_not_authorized(self):
        spec = load_and_validate()
        self.assertFalse(spec["execution_authorized"])
        self.assertEqual("draft_unapproved_unexecuted", spec["status"])
        self.assertIsNone(spec["generator"]["source_sha256"])

    def test_rejects_unauthorized_execution_and_leakage(self):
        original = load_and_validate()
        for key, value in (
            ("execution_authorized", True),
            ("status", "approved"),
            ("task_order", ["T2", "T1", "T3", "T4"]),
        ):
            changed = copy.deepcopy(original)
            changed[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_draft(changed)
        changed = copy.deepcopy(original)
        changed["data"]["test_labels_used_for_updates"] = True
        with self.assertRaises(ValueError):
            validate_draft(changed)
        changed = copy.deepcopy(original)
        changed["training_budget"]["max_hardware_calls"] = 1
        with self.assertRaises(ValueError):
            validate_draft(changed)

    def test_forgetting_scores_include_negative_improvement(self):
        matrix = [
            [0.5, 0.5, 0.5, 0.5],
            [0.8, 0.5, 0.5, 0.5],
            [0.75, 0.7, 0.5, 0.5],
            [0.7, 0.8, 0.9, 0.5],
            [0.9, 0.6, 0.75, 0.85],
        ]
        scores = forgetting_by_task(matrix)
        self.assertAlmostEqual(-0.1, scores["T1"])
        self.assertAlmostEqual(0.2, scores["T2"])
        self.assertAlmostEqual(0.15, scores["T3"])
        self.assertNotIn("T4", scores)

    def test_forgetting_rejects_incomplete_or_nonfinite(self):
        with self.assertRaises(ValueError):
            forgetting_by_task([[0.5] * 4] * 4)
        with self.assertRaises(ValueError):
            forgetting_by_task([[float("nan")] * 4] * 5)
        with self.assertRaises(ValueError):
            forgetting_by_task([[True] * 4] * 5)


if __name__ == "__main__":
    unittest.main()
