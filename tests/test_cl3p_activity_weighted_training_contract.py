"""Static arithmetic over synthetic input bits only; never model updates."""
import contextlib
import io
import json
import unittest
from dataclasses import dataclass

from integrations.continual_learning.cl2_operation_accounting import estimate_step
from integrations.continual_learning.cl2_payload_proxy import estimate_payload_proxy
from integrations.continual_learning.cl3p_activity_weighted_training import (
    prepare_training_proxy, summarize_training_inputs, main,
)


@dataclass(frozen=True)
class InputRecord:
    features: tuple[int, ...]


class CL3PActivityTrainingContracts(unittest.TestCase):
    def test_known_activity_is_summed_without_updates(self):
        records = (InputRecord((0,) * 16), InputRecord((1,) + (0,) * 15))
        summary = summarize_training_inputs(records)
        self.assertEqual(2, summary["hypothetical_update_count"])
        self.assertEqual(1, summary["active_input_bits_total"])
        self.assertEqual(0, summary["model_updates_performed"])
        for arm, totals in summary["by_arm"].items():
            profiles = (estimate_step(arm, 0), estimate_step(arm, 1))
            payloads = (estimate_payload_proxy(arm, 0), estimate_payload_proxy(arm, 1))
            self.assertEqual(sum(p.total_parameter_reads for p in profiles), totals["parameter_reads"])
            self.assertEqual(sum(p.total_parameter_writes for p in profiles), totals["parameter_writes"])
            self.assertEqual(sum(p.total_logical_payload_bytes for p in payloads), totals["logical_payload_bytes"])

    def test_fixed_five_seed_budget_and_dense_invariance(self):
        report = prepare_training_proxy()
        self.assertFalse(report["training_performed"])
        self.assertFalse(report["heldout_labels_accessed"])
        self.assertEqual(5, len(report["paired_seeds"]))
        for seed in report["paired_seeds"]:
            self.assertEqual(512, seed["hypothetical_updates_per_arm"])
            self.assertEqual(4, len(seed["tasks"]))
            for task in seed["tasks"]:
                totals = task["training_input_proxy"]
                self.assertEqual(128, totals["hypothetical_update_count"])
                self.assertEqual(128 * 4080, totals["by_arm"]["dense_sgd"]["logical_payload_bytes"])
                self.assertEqual(0, totals["model_updates_performed"])

    def test_invalid_features_and_train_cli_rejected(self):
        with self.assertRaises(ValueError):
            summarize_training_inputs((InputRecord((1,) * 15),))
        with contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(0, main(["--preview"]))
        self.assertFalse(json.loads(out.getvalue())["training_performed"])
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as err:
                main(["--train"])
        self.assertEqual(2, err.exception.code)


if __name__ == "__main__":
    unittest.main()
