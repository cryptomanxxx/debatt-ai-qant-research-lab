"""CL-5K adapter tests using only boolean decisions, never feature vectors."""
import unittest

from integrations.continual_learning.cl5j_disconnected_counter_ledger import (
    ProposalCounterLedger,
)
from integrations.continual_learning.cl5k_counter_decision_adapter import (
    account_proposed_candidate,
)


class DecisionAdapterTests(unittest.TestCase):
    def test_full_class_skips_duplicate_lookup(self):
        ledger = ProposalCounterLedger(attempt_cap=2)
        def forbidden_lookup():
            self.fail("class-full branch must short-circuit duplicate lookup")
        self.assertEqual(account_proposed_candidate(
            ledger, remaining_for_label=0, duplicate_lookup_if_open=forbidden_lookup
        ), "class_full")
        report = ledger.finish_partition(quotas_filled=True)
        self.assertEqual(report["class_full_rejections"], 1)

    def test_open_class_checks_duplicate_exactly_once(self):
        ledger = ProposalCounterLedger(attempt_cap=2)
        calls = []
        def duplicate():
            calls.append(1)
            return True
        self.assertEqual(account_proposed_candidate(
            ledger, remaining_for_label=2, duplicate_lookup_if_open=duplicate
        ), "duplicate")
        self.assertEqual(calls, [1])
        self.assertEqual(account_proposed_candidate(
            ledger, remaining_for_label=1, duplicate_lookup_if_open=lambda: False
        ), "accepted")
        report = ledger.finish_partition(quotas_filled=True)
        self.assertEqual(report["candidate_attempts"], 2)
        self.assertEqual(report["duplicate_rejections_after_class_check"], 1)
        self.assertEqual(report["accepted_records"], 1)
        self.assertFalse(report["generator_connected"])

    def test_invalid_inputs_fail_closed_before_proposal(self):
        ledger = ProposalCounterLedger()
        for quota in (True, -1, 0.0):
            with self.subTest(quota=quota), self.assertRaises(ValueError):
                account_proposed_candidate(
                    ledger, remaining_for_label=quota,
                    duplicate_lookup_if_open=lambda: False
                )
        self.assertEqual(ledger.candidate_attempts, 0)

    def test_nonboolean_duplicate_result_rejected(self):
        ledger = ProposalCounterLedger(attempt_cap=1)
        with self.assertRaises(ValueError):
            account_proposed_candidate(
                ledger, remaining_for_label=1,
                duplicate_lookup_if_open=lambda: 1
            )
        with self.assertRaises(ValueError):
            ledger.finish_partition(quotas_filled=True)


if __name__ == "__main__":
    unittest.main()
