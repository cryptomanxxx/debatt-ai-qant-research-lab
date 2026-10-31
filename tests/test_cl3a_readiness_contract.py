"""No-compute guardrail: CL-3A readiness cannot silently authorize training."""
import json
import unittest
from pathlib import Path


PATH = Path("integrations/continual_learning/cl3a_readiness.json")


class CL3AReadinessContracts(unittest.TestCase):
    def test_review_is_no_go_and_immutable_in_intent(self):
        review = json.loads(PATH.read_text(encoding="utf-8"))
        self.assertEqual("NO_GO", review["decision"])
        self.assertEqual("scientific_review_pending", review["status"])
        self.assertFalse(review["go_for_training"])
        self.assertFalse(review["compute_approved"])
        self.assertFalse(review["runner_enabled"])
        self.assertEqual(8, len(review["mandatory_blockers"]))
        self.assertTrue(all(blocker["closed"] is False for blocker in review["mandatory_blockers"]))
        self.assertIn("training", review["forbidden_actions"])
        self.assertIn("physical_energy_claim", review["forbidden_actions"])

    def test_review_proposal_agrees_with_bounded_budget(self):
        review = json.loads(PATH.read_text(encoding="utf-8"))
        plan = json.loads(Path("integrations/continual_learning/cl2_paired_run_plan_candidate.json").read_text(encoding="utf-8"))
        limits = review["candidate_limits"]
        self.assertEqual(plan["paired_seeds"], limits["seeds"])
        self.assertEqual(plan["dataset"]["train_per_task"], limits["train_per_task"])
        self.assertEqual(plan["dataset"]["test_per_task"], limits["test_per_task"])
        self.assertEqual(plan["training"]["updates_per_seed_ceiling"], limits["max_updates_per_arm_per_seed"])
        self.assertIsNone(plan["dataset"]["source_sha256"])
        self.assertFalse(plan["compute_approved"])
        self.assertFalse(plan["runner_enabled"])


if __name__ == "__main__":
    unittest.main()
