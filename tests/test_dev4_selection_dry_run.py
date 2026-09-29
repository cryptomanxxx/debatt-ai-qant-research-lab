"""Dev-4 first-round selection checks; strictly zero model training."""
import copy
import hashlib
import json
import unittest

from scripts.dev4_selection_dry_run import (
    HISTORY, PROMPT, PROTOCOL, first_round, validate_proposal, verify_inputs,
)


class SelectionDryRunTests(unittest.TestCase):
    def setUp(self):
        self.protocol = json.loads(PROTOCOL.read_text())
        self.history_bytes = HISTORY.read_bytes()
        self.prompt_bytes = PROMPT.read_bytes()
        self.history = verify_inputs(self.protocol, self.history_bytes, self.prompt_bytes)

    def test_committed_evidence_and_prompt_are_pinned(self):
        self.assertEqual(
            "sha256:" + hashlib.sha256(self.history_bytes).hexdigest(),
            self.protocol["information_policy"]["shared_initial_history_snapshot"],
        )
        self.assertEqual(
            "sha256:" + hashlib.sha256(self.prompt_bytes).hexdigest(),
            self.protocol["strategy_specifications"]["gpt-oss-120b"]["prompt_template_sha256"],
        )

    def test_first_round_never_evaluates_or_calls_groq(self):
        result = first_round(self.protocol, self.history)
        self.assertEqual(result["training_runs"], 0)
        self.assertFalse(result["evaluation_seeds_used"])
        self.assertEqual(result["selections"]["grid-search"]["local_width"], 4)
        self.assertEqual(result["selections"]["bayesian-optimization"]["local_width"], 10)
        self.assertEqual(result["selections"]["gpt-oss-120b"]["status"], "awaiting_external_response_no_api_call")
        self.assertEqual(result["selections"]["random-search"]["status"], "not_sampled_numpy_not_requested")

    def test_gpt_proposal_is_strict_and_not_executed(self):
        result = first_round(self.protocol, self.history, '{"local_width":8}')
        self.assertEqual(result["selections"]["gpt-oss-120b"]["local_width"], 8)
        self.assertEqual(result["training_runs"], 0)
        for raw in ('{"local_width":true}', '{"local_width":8.0}', '{"local_width":16}',
                    '{"local_width":8,"extra":1}', 'not json', '{"local_width":null}'):
            self.assertIsNotNone(validate_proposal(raw, list(range(4, 16)))[1])
        self.assertEqual(validate_proposal('{"local_width":8}', list(range(4, 16)), (8,)), (None, "duplicate_width"))

    def test_modified_history_and_prompt_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "history digest mismatch"):
            verify_inputs(self.protocol, self.history_bytes + b" ", self.prompt_bytes)
        with self.assertRaisesRegex(ValueError, "prompt digest mismatch"):
            verify_inputs(self.protocol, self.history_bytes, self.prompt_bytes + b" ")
        changed = copy.deepcopy(self.protocol)
        changed["guardrails"]["automatic_compute"] = True
        with self.assertRaisesRegex(ValueError, "invalid preregistration"):
            verify_inputs(changed, self.history_bytes, self.prompt_bytes)


if __name__ == "__main__":
    unittest.main()
