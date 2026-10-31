"""Only static arithmetic tests; no model or workload execution."""
import unittest

from integrations.continual_learning.cl2_operation_accounting import estimate_step
from integrations.continual_learning.cl2_payload_proxy import ProxyAssumptions, estimate_payload_proxy
from integrations.continual_learning.cl3e_measurement_ledger import (
    NOT_MEASURED, SCOPE_VERSION, measurement_scope_manifest,
    project_training_only, projection_dict,
)


class CL3EMeasurementLedgerContracts(unittest.TestCase):
    def test_training_projection_is_integer_scaled_static_scope(self):
        assumed_steps, activity = 10, 1
        one = estimate_step("local_hybrid", activity)
        payload = estimate_payload_proxy("local_hybrid", activity)
        projected = project_training_only(
            "local_hybrid", step_count=assumed_steps, active_inputs_per_step=activity
        )
        self.assertEqual(assumed_steps * one.total_parameter_reads, projected.logical_parameter_reads)
        self.assertEqual(assumed_steps * one.total_parameter_writes, projected.logical_parameter_writes)
        self.assertEqual(assumed_steps * payload.total_logical_payload_bytes, projected.logical_payload_bytes)
        self.assertEqual(SCOPE_VERSION, projected.scope_version)
        self.assertEqual(NOT_MEASURED, projected.physical_energy_joules)
        self.assertEqual(NOT_MEASURED, projection_dict(projected)["dram_bytes"])

    def test_nondefault_byte_widths_are_preserved_in_serialization(self):
        default = project_training_only(
            "sparse_shared_index", step_count=10, active_inputs_per_step=1
        )
        custom_assumptions = ProxyAssumptions(
            parameter_scalar_bytes=4, input_scalar_bytes=2, active_index_bytes=4
        )
        custom = project_training_only(
            "sparse_shared_index", step_count=10,
            active_inputs_per_step=1, assumptions=custom_assumptions
        )
        serialized = projection_dict(custom)
        self.assertEqual(4, serialized["parameter_scalar_bytes_assumed"])
        self.assertEqual(2, serialized["input_scalar_bytes_assumed"])
        self.assertEqual(4, serialized["active_index_bytes_assumed"])
        self.assertEqual(8, projection_dict(default)["parameter_scalar_bytes_assumed"])
        self.assertNotEqual(default.logical_payload_bytes, custom.logical_payload_bytes)
        self.assertEqual(
            10 * estimate_payload_proxy("sparse_shared_index", 1, custom_assumptions).total_logical_payload_bytes,
            serialized["logical_payload_bytes"],
        )

    def test_zero_and_ceiling_are_deterministic(self):
        zero = project_training_only("dense_sgd", step_count=0, active_inputs_per_step=0)
        self.assertEqual(0, zero.logical_parameter_reads)
        limit = project_training_only("sparse_shared_index", step_count=512, active_inputs_per_step=16)
        self.assertGreater(limit.logical_parameter_reads, 0)
        for count in (-1, 513, True, 1.5):
            with self.subTest(count=count), self.assertRaises(ValueError):
                project_training_only("dense_sgd", step_count=count, active_inputs_per_step=1)

    def test_separate_inference_and_physical_terms_are_explicitly_unknown(self):
        manifest = measurement_scope_manifest()
        self.assertEqual("static_predict_source_loop_projection_only_cl3n", manifest["heldout_inference_accounting"])
        self.assertEqual(NOT_MEASURED, manifest["physical_energy_joules"])
        self.assertEqual(NOT_MEASURED, manifest["data_conversion_and_optical_io"])
        self.assertFalse(manifest["trained_models"])
        self.assertFalse(manifest["experiment_authorized"])


if __name__ == "__main__":
    unittest.main()
