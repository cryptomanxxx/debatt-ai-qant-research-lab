"""CL-12 negative-only hand-authored fixtures; never generate H1 data."""
import unittest
from cl12_supplied_data_freeze_verifier import inspect_supplied_seed_records


class TestCL12(unittest.TestCase):
    def test_no_records_must_not_claim_frozen(self):
        with self.assertRaisesRegex(ValueError, "NO_GO"):
            inspect_supplied_seed_records(101, [])

    def test_reject_unplanned_seed(self):
        with self.assertRaisesRegex(ValueError, "unplanned seed"):
            inspect_supplied_seed_records(100, [])

    def test_reject_undersized_supplied_partition(self):
        example = [[0] * 16, 0]
        records = [
            {"task_id": t, "train": [example], "test": []}
            for t in ("T1", "T2", "T3", "T4")
        ]
        with self.assertRaisesRegex(ValueError, "partition size"):
            inspect_supplied_seed_records(101, records)

    def test_reject_misordered_tasks(self):
        records = [
            {"task_id": t, "train": [], "test": []}
            for t in ("T2", "T1", "T3", "T4")
        ]
        with self.assertRaisesRegex(ValueError, "task order"):
            inspect_supplied_seed_records(101, records)


if __name__ == "__main__":
    unittest.main()
