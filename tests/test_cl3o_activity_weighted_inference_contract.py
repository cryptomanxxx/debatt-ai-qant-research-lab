"""No model inference: only deterministic input-weighted cost projections."""
import contextlib
import io
import json
import unittest
from dataclasses import dataclass

from integrations.continual_learning.cl3n_inference_accounting import estimate_prediction
from integrations.continual_learning.cl3o_activity_weighted_inference import (
    prepare_activity_weighted_preview, summarize_partition_inputs, main,
)


@dataclass(frozen=True)
class Example:
    features: tuple[int, ...]
    label: int


class CL3OActivityProxyContracts(unittest.TestCase):
    def test_known_zero_and_one_active_bit_costs(self):
        examples = (
            Example((0,) * 16, 0),
            Example((1,) + (0,) * 15, 1),
        )
        summary = summarize_partition_inputs(examples)
        self.assertEqual(2, summary["input_examples"])
        self.assertEqual(1, summary["active_input_bits_total"])
        self.assertEqual(1 / 32, summary["input_activity_fraction"])
        self.assertEqual(0, summary["training_or_predict_calls"])
        for arm, totals in summary["by_arm"].items():
            first = estimate_prediction(arm, 0)
            second = estimate_prediction(arm, 1)
            self.assertEqual(
                first.logical_payload_bytes + second.logical_payload_bytes,
                totals["logical_payload_bytes"],
            )
            self.assertEqual(0, totals["parameter_writes"])

    def test_five_seed_four_task_preview_has_no_model_results(self):
        report = prepare_activity_weighted_preview()
        self.assertEqual(5, len(report["paired_seeds"]))
        self.assertFalse(report["training_performed"])
        self.assertFalse(report["model_predictions_performed"])
        self.assertFalse(report["physical_energy_measured"])
        for seed in report["paired_seeds"]:
            self.assertEqual(64, len(seed["dataset_content_sha256"]))
            self.assertEqual(4, len(seed["tasks"]))
            for task in seed["tasks"]:
                self.assertEqual(128, task["train_input_preview"]["input_examples"])
                self.assertEqual(64, task["heldout_input_preview"]["input_examples"])
                self.assertEqual(
                    128 * 1376,
                    task["train_input_preview"]["by_arm"]["dense_sgd"]["logical_payload_bytes"],
                )
                self.assertEqual(
                    task["heldout_input_preview"]["by_arm"]["sparse_local_hybrid"],
                    task["heldout_input_preview"]["by_arm"]["sparse_shared_index"],
                )

    def test_invalid_input_or_cli_rejected(self):
        with self.assertRaises(ValueError):
            summarize_partition_inputs((Example((0,) * 15, 0),))
        with contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(0, main(["--preview"]))
        self.assertFalse(json.loads(out.getvalue())["model_predictions_performed"])
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as e:
                main(["--train"])
        self.assertEqual(2, e.exception.code)


if __name__ == "__main__":
    unittest.main()
