"""CL-2 candidate data contract tests only. These do not train or score models."""
import unittest

from integrations.continual_learning.cl2_data_candidate import (
    ACTIVITY_RATES,
    INPUT_SIZE,
    TASK_PAIRS,
    dataset_sha256,
    generate_candidate,
)


class CL2DataCandidateTests(unittest.TestCase):
    def test_deterministic_paired_seeds_and_order(self):
        first = generate_candidate(101)
        second = generate_candidate(101)
        different = generate_candidate(102)
        self.assertEqual(first, second)
        self.assertEqual(dataset_sha256(first), dataset_sha256(second))
        self.assertNotEqual(dataset_sha256(first), dataset_sha256(different))
        self.assertEqual(tuple(TASK_PAIRS), tuple(x.task_id for x in first))
        self.assertEqual(64, len(dataset_sha256(first)))

    def test_balanced_heldout_and_global_disjointness(self):
        splits = generate_candidate(101)
        seen = set()
        for task in splits:
            self.assertEqual(128, len(task.train))
            self.assertEqual(64, len(task.test))
            for partition in (task.train, task.test):
                self.assertEqual(len(partition) // 2, sum(e.label for e in partition))
                for example in partition:
                    self.assertEqual(INPUT_SIZE, len(example.features))
                    self.assertTrue(set(example.features) <= {0, 1})
                    self.assertEqual(
                        example.features[TASK_PAIRS[task.task_id][0]]
                        ^ example.features[TASK_PAIRS[task.task_id][1]],
                        example.label,
                    )
                    self.assertNotIn(example.features, seen)
                    seen.add(example.features)
        self.assertEqual(4 * 192, len(seen))
        self.assertEqual(4, len(ACTIVITY_RATES))

    def test_rejects_invalid_seeds_and_sizes(self):
        for seed in (-1, True, 1.2):
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                generate_candidate(seed)
        for size in (0, 3, True):
            with self.subTest(size=size), self.assertRaises(ValueError):
                generate_candidate(101, train_per_task=size)


if __name__ == "__main__":
    unittest.main()
