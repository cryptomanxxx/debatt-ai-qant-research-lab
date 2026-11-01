"""CL-5D exact combinatorics tests; zero dataset generation."""
import unittest

from integrations.continual_learning.cl5d_xor_support_audit import (
    planned_quota_capacity,
    xor_support_by_weight,
)


class XorSupportTests(unittest.TestCase):
    def test_support_conservation(self):
        audit = xor_support_by_weight()
        self.assertEqual(audit["support_per_class"], {"0": 32768, "1": 32768})
        self.assertEqual(len(audit["weights"]), 17)
        self.assertEqual(sum(x["total_support"] for x in audit["weights"]), 65536)
        self.assertEqual(audit["observed_samples"], 0)

    def test_low_and_high_hamming_weights(self):
        rows = xor_support_by_weight()["weights"]
        self.assertEqual((rows[0]["label_zero_support"], rows[0]["label_one_support"]), (1, 0))
        self.assertEqual((rows[1]["label_zero_support"], rows[1]["label_one_support"]), (14, 2))
        self.assertEqual((rows[16]["label_zero_support"], rows[16]["label_one_support"]), (1, 0))

    def test_nominal_quota_is_necessary_only(self):
        audit = planned_quota_capacity()
        self.assertEqual(audit["required_distinct_per_class_across_all_tasks"], 384)
        self.assertEqual(audit["available_distinct_per_class"], 32768)
        self.assertFalse(audit["sampling_feasibility_proven"])
        self.assertEqual(audit["decision"], "NO_GO")

    def test_fail_closed_over_capacity_and_bad_counts(self):
        with self.assertRaises(ValueError):
            planned_quota_capacity(train_per_task=20000, test_per_task=20000)
        for values in ((1, 64), (128, 3), (0, 64), (True, 64), (128.0, 64)):
            with self.subTest(values=values), self.assertRaises(ValueError):
                planned_quota_capacity(train_per_task=values[0], test_per_task=values[1])


if __name__ == "__main__":
    unittest.main()
