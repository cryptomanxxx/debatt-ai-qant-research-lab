"""No-model-training contracts for a noncontradictory CL-2 primary candidate."""
import unittest

from integrations.continual_learning.cl2_data_candidate import INPUT_SIZE
from integrations.continual_learning.cl2_shared_rule_candidate import (
    generate_shared_rule_candidate,
    global_label,
    shared_rule_dataset_sha256,
)


class CL2SharedRuleCandidateTests(unittest.TestCase):
    def test_paired_seed_reproducibility_and_no_overlap(self):
        a = generate_shared_rule_candidate(101)
        b = generate_shared_rule_candidate(101)
        self.assertEqual(a, b)
        self.assertEqual(shared_rule_dataset_sha256(a), shared_rule_dataset_sha256(b))
        self.assertNotEqual(
            shared_rule_dataset_sha256(a),
            shared_rule_dataset_sha256(generate_shared_rule_candidate(102)),
        )
        self.assertEqual(("T1", "T2", "T3", "T4"), tuple(task.task_id for task in a))
        seen = set()
        for task in a:
            self.assertEqual(128, len(task.train))
            self.assertEqual(64, len(task.test))
            for partition in (task.train, task.test):
                self.assertEqual(len(partition) // 2, sum(row.label for row in partition))
                for row in partition:
                    self.assertEqual(INPUT_SIZE, len(row.features))
                    self.assertEqual(global_label(row.features), row.label)
                    self.assertNotIn(row.features, seen)
                    seen.add(row.features)
        self.assertEqual(768, len(seen))

    def test_same_vector_has_same_label_for_every_task(self):
        samples = generate_shared_rule_candidate(101)
        for task in samples:
            for row in task.test[:10]:
                self.assertEqual(global_label(row.features), row.label)
        witness = (1, 0) + (0,) * 14
        self.assertEqual(1, global_label(witness))

    def test_rejects_bad_sizes_and_features(self):
        for invalid in (True, -1, 1.0):
            with self.subTest(seed=invalid), self.assertRaises(ValueError):
                generate_shared_rule_candidate(invalid)
        with self.assertRaises(ValueError):
            generate_shared_rule_candidate(101, test_per_task=3)
        with self.assertRaises(ValueError):
            global_label((0, 1))
        with self.assertRaises(ValueError):
            global_label((2,) + (0,) * 15)


if __name__ == "__main__":
    unittest.main()
