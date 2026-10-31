"""No-training checks for the candidate CL-2 paired protocol."""
import copy
import unittest

from integrations.continual_learning.cl2_paired_run_plan import (
    load_candidate,
    validate_candidate,
)


class PairedPlanContracts(unittest.TestCase):
    def test_plan_explicitly_not_executable(self):
        plan = load_candidate()
        self.assertEqual(5, len(plan["paired_seeds"]))
        self.assertEqual(154, plan["models"][0]["parameters"])
        self.assertEqual(154, plan["models"][1]["parameters"])
        self.assertEqual(512, plan["training"]["updates_per_seed_ceiling"])
        self.assertFalse(plan["compute_approved"])
        self.assertIsNone(plan["dataset"]["source_sha256"])

    def test_rejects_premature_approval_and_changed_science(self):
        original = load_candidate()
        mutations = (
            ("compute_approved", True),
            ("runner_enabled", True),
            ("paired_seeds", [1, 2, 3]),
        )
        for key, value in mutations:
            altered = copy.deepcopy(original)
            altered[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_candidate(altered)
        altered = copy.deepcopy(original)
        altered["dataset"]["task_id_supplied_to_model"] = True
        with self.assertRaises(ValueError):
            validate_candidate(altered)
        altered = copy.deepcopy(original)
        altered["evaluation"]["heldout_updates_allowed"] = True
        with self.assertRaises(ValueError):
            validate_candidate(altered)
        altered = copy.deepcopy(original)
        altered["models"][1]["parameters"] = 155
        with self.assertRaises(ValueError):
            validate_candidate(altered)

    def test_no_pseudophysical_energy_claim(self):
        altered = load_candidate()
        altered["interpretation_gates"]["energy_savings_claims_allowed"] = True
        with self.assertRaises(ValueError):
            validate_candidate(altered)


if __name__ == "__main__":
    unittest.main()
