"""Dev-4 selection policy adapters: no evaluation or network."""
import json
import unittest

from scripts.dev4_policy_adapters import (
    choose_bayesian, choose_gpt_external, choose_grid, choose_random, random_stream,
)
from scripts.dev4_selection_dry_run import PROTOCOL


class PolicyAdapterTests(unittest.TestCase):
    def setUp(self):
        self.p = json.loads(PROTOCOL.read_text())

    def test_grid_and_random_are_five_unique_preregistered_draws(self):
        self.assertEqual([choose_grid(self.p, i) for i in range(1, 6)], [4, 5, 6, 7, 8])
        draws = random_stream(self.p)
        self.assertEqual(len(draws), 5)
        self.assertEqual(len(set(draws)), 5)
        self.assertEqual([choose_random(self.p, i) for i in range(1, 6)], draws)
        with self.assertRaises(ValueError):
            choose_random(self.p, 6)

    def test_bayesian_fixed_initialization_and_failed_round_fallback(self):
        self.assertEqual(choose_bayesian(self.p, [], []), 10)
        ledger = [{"round": 1, "local_width": 10, "valid": True, "status": "evaluation_failed"}]
        self.assertEqual(choose_bayesian(self.p, ledger, []), 4)

    def test_bayesian_fits_only_own_valid_paired_observations(self):
        ledger = [{"round": 1, "local_width": 10, "valid": True, "status": "evaluated"}]
        outcomes = [{"round": 1, "paired_outcome": {
            "candidate_correct": 420, "control_correct": 425,
            "candidate_parameters": 20, "control_parameters": 30,
        }}]
        choice = choose_bayesian(self.p, ledger, outcomes)
        self.assertIn(choice, self.p["search_space"]["candidate_local_widths"])
        self.assertNotEqual(choice, 10)
        self.assertEqual(choice, choose_bayesian(self.p, ledger, outcomes))
        with self.assertRaisesRegex(ValueError, "mismatch"):
            choose_bayesian(self.p, ledger, [])
        with self.assertRaisesRegex(ValueError, "completed outcome ledger mismatch"):
            choose_bayesian(self.p, ledger, outcomes + [{"round": 99, "paired_outcome": outcomes[0]["paired_outcome"]}])

    def test_external_gpt_rejects_duplicate_even_after_failure(self):
        ledger = [{"round": 1, "local_width": 8, "valid": True, "status": "evaluation_failed"}]
        self.assertEqual(choose_gpt_external(self.p, '{"local_width":8}', ledger), (None, "duplicate_width"))
        self.assertEqual(choose_gpt_external(self.p, '{"local_width":9}', ledger), (9, None))


if __name__ == "__main__":
    unittest.main()
