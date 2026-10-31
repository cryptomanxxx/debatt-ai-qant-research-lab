"""Only separately scoped arithmetic projections, no actual AI training."""
import contextlib
import io
import json
import unittest

from integrations.continual_learning.cl3q_joint_cost_review import (
    compare_static_proxies, main,
)


class CL3QJointCostContracts(unittest.TestCase):
    def test_per_seed_comparison_keeps_training_and_inference_separate(self):
        report = compare_static_proxies()
        self.assertEqual(5, len(report["paired_seeds"]))
        self.assertFalse(report["model_training_or_inference_performed"])
        self.assertFalse(report["accuracy_measured"])
        self.assertFalse(report["physical_energy_measured"])
        self.assertFalse(report["cost_ratios_are_energy_ratios"])
        for seed in report["paired_seeds"]:
            self.assertEqual(512, seed["hypothetical_training_updates_per_arm"])
            self.assertEqual(256, seed["heldout_feature_count_per_arm"])
            dense = seed["by_arm"]["dense_sgd"]
            self.assertEqual(512 * 4080, dense["training_logical_payload_bytes"])
            self.assertEqual(256 * 1376, dense["heldout_inference_logical_payload_bytes"])
            self.assertEqual(0, dense["heldout_inference_logical_parameter_writes"])
            self.assertEqual(1.0, dense["training_payload_ratio_to_dense"])
            self.assertEqual(1.0, dense["heldout_inference_payload_ratio_to_dense"])
            self.assertEqual(
                seed["by_arm"]["sparse_local_hybrid"]["heldout_inference_logical_payload_bytes"],
                seed["by_arm"]["sparse_shared_index"]["heldout_inference_logical_payload_bytes"],
            )

    def test_no_experiment_execution_cli(self):
        with contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(0, main(["--preview"]))
        report = json.loads(out.getvalue())
        self.assertFalse(report["acceptable_accuracy_tradeoff_verified"])
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as ctx:
                main(["--execute"])
        self.assertEqual(2, ctx.exception.code)


if __name__ == "__main__":
    unittest.main()
