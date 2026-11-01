"""CL-5J event-only unit tests, without synthetic feature vectors."""
import unittest

from integrations.continual_learning.cl5j_disconnected_counter_ledger import (
    ProposalCounterLedger,
)


class DisconnectedLedgerTests(unittest.TestCase):
    def test_balanced_event_accounting_and_class_priority(self):
        ledger = ProposalCounterLedger(attempt_cap=4)
        ledger.begin_proposal()
        self.assertEqual(ledger.resolve_proposal(class_full=True,
                         duplicate_if_class_open=None), "class_full")
        ledger.begin_proposal()
        self.assertEqual(ledger.resolve_proposal(class_full=False,
                         duplicate_if_class_open=True), "duplicate")
        ledger.begin_proposal()
        self.assertEqual(ledger.resolve_proposal(class_full=False,
                         duplicate_if_class_open=False), "accepted")
        result = ledger.finish_partition(quotas_filled=True)
        self.assertEqual(result["candidate_attempts"], 3)
        self.assertEqual(result["accepted_records"], 1)
        self.assertEqual(result["class_full_rejections"], 1)
        self.assertEqual(result["duplicate_rejections_after_class_check"], 1)
        self.assertFalse(result["generator_connected"])
        self.assertFalse(result["dataset_generation_executed"])
        self.assertEqual(result["scientific_decision"], "NO_GO")

    def test_cap_exhaustion_without_completion(self):
        ledger = ProposalCounterLedger(attempt_cap=1)
        ledger.begin_proposal()
        ledger.resolve_proposal(class_full=True, duplicate_if_class_open=None)
        result = ledger.finish_partition(quotas_filled=False)
        self.assertTrue(result["attempt_cap_exhausted"])
        with self.assertRaises(ValueError):
            ledger.begin_proposal()

    def test_fail_closed_on_malformed_events(self):
        ledger = ProposalCounterLedger(attempt_cap=2)
        with self.assertRaises(ValueError):
            ledger.resolve_proposal(class_full=False, duplicate_if_class_open=False)
        ledger.begin_proposal()
        with self.assertRaises(ValueError):
            ledger.begin_proposal()
        with self.assertRaises(ValueError):
            ledger.resolve_proposal(class_full=True, duplicate_if_class_open=True)
        with self.assertRaises(ValueError):
            ledger.finish_partition(quotas_filled=True)
        ledger.resolve_proposal(class_full=True, duplicate_if_class_open=None)
        with self.assertRaises(ValueError):
            ledger.finish_partition(quotas_filled=False)

    def test_reject_invalid_cap(self):
        for bad in (0, -1, True, 2.0):
            with self.subTest(value=bad), self.assertRaises(ValueError):
                ProposalCounterLedger(attempt_cap=bad)


if __name__ == "__main__":
    unittest.main()
