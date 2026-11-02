"""CL-10G contract tests with fabricated binary fixtures, not H1 datasets."""
import copy
import unittest

from cl10g_train_only_split import (
    TASKS, plan_train_only_calibration,
)


def schema_fixtures():
    """Four small synthetic grid shapes expanded solely to test the contract."""
    result = {}
    for task_no, task in enumerate(TASKS):
        rows = []
        for label in (0, 1):
            for index in range(64):
                number = task_no * 128 + label * 64 + index
                tail = tuple((number >> bit) & 1 for bit in range(13, -1, -1))
                rows.append(((0, label) + tail, label))
        result[task] = rows
    return result


class TestCL10G(unittest.TestCase):
    def test_full_fixture_is_deterministic_disjoint_and_train_only(self):
        rows = schema_fixtures()
        result = plan_train_only_calibration(101, rows)
        self.assertEqual(result, plan_train_only_calibration(101, rows))
        self.assertEqual(result["decision"], "NO_GO")
        self.assertFalse(result["heldout_accessed"])
        self.assertFalse(result["training_authorized"])
        for item in result["tasks"]:
            fit = set(item["fit_indices"])
            valid = set(item["validation_indices"])
            self.assertEqual(len(fit), 96)
            self.assertEqual(len(valid), 32)
            self.assertFalse(fit & valid)
            self.assertEqual(fit | valid, set(range(128)))
            original = rows[item["task"]]
            self.assertEqual(sum(original[i][1] for i in valid), 16)
            self.assertEqual(sum(original[i][1] for i in fit), 48)

    def test_input_order_is_bound_to_digest(self):
        rows = schema_fixtures()
        before = plan_train_only_calibration(101, rows)
        swapped = copy.deepcopy(rows)
        swapped["T1"][0], swapped["T1"][1] = swapped["T1"][1], swapped["T1"][0]
        after = plan_train_only_calibration(101, swapped)
        self.assertNotEqual(before["tasks"][0]["training_order_digest"],
                            after["tasks"][0]["training_order_digest"])
        self.assertEqual(before["tasks"][0]["validation_indices"] !=
                         after["tasks"][0]["validation_indices"], True)

    def test_fail_closed_on_leakage_and_quota_drift(self):
        changes = []
        rows = schema_fixtures()
        rows["T2"][0] = rows["T1"][0]
        changes.append(rows)
        rows = schema_fixtures()
        rows["T1"][0] = (rows["T1"][0][0], 1)
        changes.append(rows)
        rows = schema_fixtures()
        rows["T1"].pop()
        changes.append(rows)
        rows = schema_fixtures()
        rows["heldout"] = rows["T1"]
        changes.append(rows)
        for rows in changes:
            with self.subTest(keys=list(rows)), self.assertRaises(ValueError):
                plan_train_only_calibration(101, rows)

    def test_wrong_seed_or_unexpected_features_rejected(self):
        with self.assertRaises(ValueError):
            plan_train_only_calibration(999, schema_fixtures())
        rows = schema_fixtures()
        rows["T1"][0] = ((0, 0, 2) + rows["T1"][0][0][3:], 0)
        with self.assertRaises(ValueError):
            plan_train_only_calibration(101, rows)


if __name__ == "__main__":
    unittest.main()
