"""Analytical XOR data controls, never model training or inference."""
import contextlib
import io
import json
import unittest
from dataclasses import dataclass

from integrations.continual_learning.cl3m_reference_controls import (
    main, prepare_references, reference_partition,
)


@dataclass(frozen=True)
class Example:
    features: tuple[int, ...]
    label: int


class CL3MReferenceContracts(unittest.TestCase):
    def test_known_xor_baselines_are_defined_by_data_not_a_model(self):
        report = prepare_references()
        self.assertFalse(report["training_performed"])
        self.assertFalse(report["experiment_authorized"])
        self.assertEqual([101, 102, 103, 104, 105],
                         [row["seed"] for row in report["paired_seeds"]])
        for seed in report["paired_seeds"]:
            self.assertEqual(4, len(seed["tasks"]))
            for task in seed["tasks"]:
                for partition, n in (("train", 128), ("test", 64)):
                    row = task[partition]
                    self.assertEqual(n, row["count"])
                    self.assertEqual(0.5, row["always_predict_zero_accuracy"])
                    self.assertEqual(0.5, row["always_predict_one_accuracy"])
                    self.assertEqual(1.0, row["known_xor_rule_accuracy"])
                    self.assertEqual("NOT_MEASURED", row["learned_model_accuracy"])

    def test_rejects_imbalanced_and_wrong_rule_fixture(self):
        zero = Example((0,) * 16, 0)
        one = Example((1, 0) + (0,) * 14, 1)
        self.assertEqual(1.0, reference_partition((zero, one))["known_xor_rule_accuracy"])
        with self.assertRaises(ValueError):
            reference_partition((zero, zero))
        with self.assertRaises(ValueError):
            reference_partition((zero, Example(one.features, 0)))

    def test_cli_has_no_train_mode(self):
        with contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(0, main(["--check-data"]))
        self.assertFalse(json.loads(out.getvalue())["training_performed"])
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as ctx:
                main(["--train"])
        self.assertEqual(2, ctx.exception.code)


if __name__ == "__main__":
    unittest.main()
