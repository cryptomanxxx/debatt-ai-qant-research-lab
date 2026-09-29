"""Strict zero-compute five-round replay checks."""
import copy
import json
import unittest

from scripts.dev4_five_round_replay import STRATEGIES, replay
from scripts.dev4_selection_dry_run import PROTOCOL


def records():
    rows = []
    for number in range(1, 6):
        for strategy in STRATEGIES:
            rows.append({
                "round": number, "strategy": strategy,
                "proposal": json.dumps({"local_width": number + 3}),
                "status": "evaluated",
                "paired_outcome": {
                    "candidate_correct": 420, "control_correct": 425,
                    "candidate_parameters": 20, "control_parameters": 30,
                },
            })
    return rows


class ReplayTests(unittest.TestCase):
    def setUp(self):
        self.protocol = json.loads(PROTOCOL.read_text())

    def test_five_rounds_no_compute_and_only_prior_own_context(self):
        result = replay(self.protocol, records())
        self.assertEqual(len(result["audit"]), 20)
        self.assertEqual((result["training_runs"], result["groq_calls"], result["holdout_access"]), (0, 0, False))
        for i, row in enumerate(result["audit"]):
            self.assertEqual(row["prior_own_attempts"], i // 4)
            self.assertEqual(row["prior_own_completed_outcomes"], i // 4)

    def test_failed_and_invalid_attempts_consume_round_without_fake_outcomes(self):
        data = records()
        data[0]["status"] = "evaluation_failed"
        data[0]["paired_outcome"] = None
        data[4]["proposal"] = '{"local_width":4}'  # same width failed in round 1
        data[4]["status"] = "invalid"
        data[4]["paired_outcome"] = None
        result = replay(self.protocol, data)
        self.assertEqual(result["audit"][4]["proposal_error"], "duplicate_width")
        self.assertEqual(result["audit"][8]["prior_own_attempts"], 2)
        self.assertEqual(result["audit"][8]["prior_own_completed_outcomes"], 0)
        data[0]["paired_outcome"] = {"candidate_correct": 500}
        with self.assertRaisesRegex(ValueError, "failed evaluation"):
            replay(self.protocol, data)

    def test_no_partial_or_unordered_replay(self):
        data = records()
        with self.assertRaisesRegex(ValueError, "exactly five rounds"):
            replay(self.protocol, data[:-1])
        data[0], data[1] = data[1], data[0]
        with self.assertRaisesRegex(ValueError, "ordered"):
            replay(self.protocol, data)

    def test_outcomes_are_strictly_paired(self):
        data = records()
        data[0]["paired_outcome"]["future_holdout"] = 999
        with self.assertRaisesRegex(ValueError, "exactly four fields"):
            replay(self.protocol, data)
        data = records()
        data[0]["paired_outcome"]["candidate_correct"] = 501
        with self.assertRaisesRegex(ValueError, "exceeds evaluation budget"):
            replay(self.protocol, data)
        data = records()
        data[0]["paired_outcome"]["candidate_correct"] = True
        with self.assertRaisesRegex(ValueError, "nonnegative integers"):
            replay(self.protocol, data)


if __name__ == "__main__":
    unittest.main()
