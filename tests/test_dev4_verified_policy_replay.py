"""Policy-verified five-round transcript tests; fixture metrics are NOT evidence."""
import copy
import json
import unittest

from scripts.dev4_five_round_replay import STRATEGIES, replay
from scripts.dev4_policy_adapters import choose_bayesian, choose_grid, choose_random
from scripts.dev4_selection_dry_run import PROTOCOL


def fixture(protocol):
    """Synthetic unit-test fixture only; never publish as benchmark results."""
    records, bo_ledger, bo_outcomes = [], [], []
    paired = {"candidate_correct": 420, "control_correct": 425,
              "candidate_parameters": 20, "control_parameters": 30}
    for number in range(1, 6):
        for strategy in STRATEGIES:
            if strategy == "grid-search":
                width = choose_grid(protocol, number)
            elif strategy == "random-search":
                width = choose_random(protocol, number)
            elif strategy == "bayesian-optimization":
                width = choose_bayesian(protocol, bo_ledger, bo_outcomes)
                bo_ledger.append({"round": number, "local_width": width,
                                  "valid": True, "status": "evaluated"})
                bo_outcomes.append({"round": number, "paired_outcome": dict(paired)})
            else:
                width = number + 3
            records.append({"round": number, "strategy": strategy,
                            "proposal": json.dumps({"local_width": width}),
                            "status": "evaluated", "paired_outcome": dict(paired)})
    return records


class VerifiedPolicyReplayTests(unittest.TestCase):
    def setUp(self):
        self.protocol = json.loads(PROTOCOL.read_text())

    def test_five_round_policy_verified_transcript(self):
        result = replay(self.protocol, fixture(self.protocol), verify_policies=True)
        self.assertTrue(result["policies_verified"])
        self.assertEqual(len(result["audit"]), 20)
        self.assertEqual((result["training_runs"], result["groq_calls"], result["holdout_access"]),
                         (0, 0, False))
        for index, entry in enumerate(result["audit"]):
            self.assertEqual(entry["prior_own_attempts"], index // 4)
            self.assertEqual(entry["prior_own_completed_outcomes"], index // 4)

    def test_grid_and_random_tampering_is_rejected(self):
        for index in (1, 2):
            rows = fixture(self.protocol)
            rows[index]["proposal"] = '{"local_width":14}'
            with self.subTest(index=index), self.assertRaisesRegex(ValueError, "policy selection mismatch"):
                replay(self.protocol, rows, verify_policies=True)

    def test_bayesian_cannot_use_a_forged_selection(self):
        rows = fixture(self.protocol)
        rows[3]["proposal"] = '{"local_width":11}'  # fixed first BO choice is 10
        with self.assertRaisesRegex(ValueError, "policy selection mismatch"):
            replay(self.protocol, rows, verify_policies=True)

    def test_bayesian_reacts_only_to_own_prior_completed_outcomes(self):
        rows = fixture(self.protocol)
        # Another strategy's prior score can change, but BO's selections cannot.
        changed = copy.deepcopy(rows)
        changed[0]["paired_outcome"]["candidate_correct"] = 1
        original = replay(self.protocol, rows, verify_policies=True)
        modified = replay(self.protocol, changed, verify_policies=True)
        self.assertEqual(
            [x["local_width"] for x in original["audit"] if x["strategy"] == "bayesian-optimization"],
            [x["local_width"] for x in modified["audit"] if x["strategy"] == "bayesian-optimization"],
        )

    def test_missing_evaluation_still_fails_closed(self):
        rows = fixture(self.protocol)
        rows[3]["paired_outcome"] = None
        with self.assertRaisesRegex(ValueError, "paired outcome"):
            replay(self.protocol, rows, verify_policies=True)


if __name__ == "__main__":
    unittest.main()
