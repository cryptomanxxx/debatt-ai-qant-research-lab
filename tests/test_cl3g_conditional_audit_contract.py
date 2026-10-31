"""Only synthetic distribution audits; never trains or scores a model."""
import contextlib
import io
import json
import unittest
from dataclasses import dataclass

from integrations.continual_learning.cl3g_conditional_audit import (
    audit_conditional_seed, conditional_partition, main, prepare_conditional_audit,
)


@dataclass(frozen=True)
class Row:
    features: tuple[int, ...]
    label: int


class CL3GConditionalAuditContracts(unittest.TestCase):
    def test_class_specific_activity_uses_correct_denominators(self):
        zero = Row((0,) * 16, 0)
        one = Row((1, 0) + (0,) * 14, 1)
        result = conditional_partition((zero, one))
        self.assertEqual(1, result["by_label"]["0"]["count"])
        self.assertEqual(1, result["by_label"]["1"]["count"])
        self.assertEqual(0.0, result["by_label"]["0"]["activity_fraction"])
        self.assertEqual(1 / 16, result["by_label"]["1"]["activity_fraction"])
        self.assertEqual(0.5, result["by_label"]["1"]["label_pair_activity_fraction"])
        self.assertAlmostEqual(1 / 16, result["absolute_conditional_activity_gap"])

    def test_all_five_paired_seed_distributions_are_reproducible(self):
        report = prepare_conditional_audit()
        self.assertFalse(report["training_performed"])
        self.assertFalse(report["experiment_authorized"])
        self.assertEqual([101, 102, 103, 104, 105],
                         [seed["seed"] for seed in report["paired_seeds"]])
        for seed in report["paired_seeds"]:
            self.assertEqual(4, len(seed["tasks"]))
            for task in seed["tasks"]:
                for partition, count in (("train", 64), ("test", 32)):
                    by_label = task[partition]["by_label"]
                    self.assertEqual(count, by_label["0"]["count"])
                    self.assertEqual(count, by_label["1"]["count"])
                    self.assertTrue(0 <= task[partition]["absolute_conditional_activity_gap"] <= 1)
        self.assertEqual(report["paired_seeds"][0], audit_conditional_seed(101))

    def test_rejects_missing_class_or_invalid_command(self):
        with self.assertRaises(ValueError):
            conditional_partition((Row((0,) * 16, 0),))
        with contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(0, main(["--audit"]))
        self.assertFalse(json.loads(out.getvalue())["training_performed"])
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as caught:
                main(["--train"])
        self.assertEqual(2, caught.exception.code)


if __name__ == "__main__":
    unittest.main()
