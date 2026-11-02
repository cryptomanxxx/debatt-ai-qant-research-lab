"""CL-10B no-training tests: only static scope and rejection semantics."""
import copy
import json
import unittest
from pathlib import Path
from unittest.mock import patch

from cl10b_gated_h1_runner import (
    ExecutionNotAuthorized, PROPOSAL, main, validate_review_only_proposal,
)


class TestCL10B(unittest.TestCase):
    def setUp(self):
        self.proposal = json.loads(PROPOSAL.read_text(encoding="utf-8"))

    def test_proposed_scope_no_go(self):
        result = validate_review_only_proposal(self.proposal)
        self.assertEqual(result["decision"], "NO_GO")
        self.assertEqual(result["models_constructed"], 0)
        self.assertEqual(result["datasets_generated"], 0)

    def test_cli_refuses_training(self):
        with patch("builtins.print"):
            with self.assertRaises(ExecutionNotAuthorized):
                main()

    def test_scope_and_authorization_mutations_rejected(self):
        changes = (
            lambda x: x.update(decision="GO"),
            lambda x: x.update(training_authorized=True),
            lambda x: x.update(inference_authorized=True),
            lambda x: x.update(dataset_generation_authorized=True),
            lambda x: x.update(runnable_workflow_created=True),
            lambda x: x["training_plan"].update(seeds=[101]),
            lambda x: x["training_plan"].update(total_updates_max=999999),
            lambda x: x["training_plan"].update(automatic_retries=True),
            lambda x: x["dataset"].update(heldout_per_task=128),
            lambda x: x["evaluation"].update(checkpoints=["post_T4"]),
        )
        for mutate in changes:
            with self.subTest(mutation=str(mutate)):
                changed = copy.deepcopy(self.proposal)
                mutate(changed)
                with self.assertRaises(ValueError):
                    validate_review_only_proposal(changed)


if __name__ == "__main__":
    unittest.main()
