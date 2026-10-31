"""Static dataset-design assertions. No model fit or research result generation."""
import unittest

from integrations.continual_learning.cl2_data_candidate import (
    Example,
    TaskSplit,
    generate_candidate,
)
from integrations.continual_learning.cl2_design_audit import (
    audit_candidate_splits,
    exact_uniform_rule_conflict,
)


class CL2DesignAuditTests(unittest.TestCase):
    def test_observed_activity_and_balance_are_measured(self):
        splits = generate_candidate(101)
        report = audit_candidate_splits(splits)
        self.assertEqual(8, len(report))
        self.assertEqual(4, len({r.task_id for r in report}))
        self.assertEqual(("train", "test"), (report[0].partition, report[1].partition))
        for row in report:
            self.assertEqual(0.5, row.positive_fraction)
            self.assertGreater(row.observed_spike_fraction, 0)
            self.assertLess(row.observed_spike_fraction, 1)
            self.assertIn(row.examples, (128, 64))

    def test_rules_incompatible_without_task_identity(self):
        for a in ("T1", "T2", "T3", "T4"):
            self.assertEqual(0.0, exact_uniform_rule_conflict(a, a))
            for b in ("T1", "T2", "T3", "T4"):
                if a != b:
                    self.assertEqual(0.5, exact_uniform_rule_conflict(a, b))
        with self.assertRaises(ValueError):
            exact_uniform_rule_conflict("T5", "T1")

    def test_leakage_or_corrupted_label_rejected(self):
        originals = generate_candidate(101)
        first = originals[0]
        corrupted = (TaskSplit(first.task_id, first.train, first.train[:1] + first.test[1:]),) + originals[1:]
        with self.assertRaises(ValueError):
            audit_candidate_splits(corrupted)
        bad_example = Example(first.train[0].features, 1 - first.train[0].label)
        corrupted = (
            TaskSplit(first.task_id, (bad_example,) + first.train[1:], first.test),
        ) + originals[1:]
        with self.assertRaises(ValueError):
            audit_candidate_splits(corrupted)


if __name__ == "__main__":
    unittest.main()
