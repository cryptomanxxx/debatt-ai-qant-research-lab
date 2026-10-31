"""Pure inference source-loop estimates only; no trained model or prediction."""
import unittest

from integrations.continual_learning.cl2_payload_proxy import ProxyAssumptions
from integrations.continual_learning.cl3n_inference_accounting import estimate_prediction


class CL3NInferenceAccountingContracts(unittest.TestCase):
    def test_dense_and_inherited_local_have_identical_prediction_scope(self):
        for active in (0, 1, 8, 16):
            dense = estimate_prediction("dense_sgd", active)
            local = estimate_prediction("local_hybrid", active)
            self.assertEqual(dense.parameter_reads, local.parameter_reads)
            self.assertEqual(154, dense.parameter_reads)
            self.assertEqual(144, dense.forward_weight_terms)
            self.assertEqual(144, dense.input_feature_visits)
            self.assertEqual(0, dense.parameter_writes)
            self.assertEqual(1376, dense.logical_payload_bytes)

    def test_both_sparse_predict_paths_are_equal_not_step_only_optimized(self):
        for active in (0, 1, 8, 16):
            a = estimate_prediction("sparse_local_hybrid", active)
            b = estimate_prediction("sparse_shared_index", active)
            self.assertEqual(a, type(a)(**{**a.__dict__, "arm": b.arm}))
            self.assertEqual(16 + active * 8, a.forward_weight_terms)
            self.assertEqual(32, a.input_feature_visits)
            self.assertEqual(9 * active, a.active_index_reads + a.active_index_writes)
            self.assertEqual(0, a.parameter_writes)
        self.assertLess(
            estimate_prediction("sparse_shared_index", 1).logical_payload_bytes,
            estimate_prediction("dense_sgd", 1).logical_payload_bytes,
        )
        self.assertGreater(
            estimate_prediction("sparse_shared_index", 16).logical_payload_bytes,
            estimate_prediction("dense_sgd", 16).logical_payload_bytes,
        )

    def test_nondefault_widths_and_invalid_activity(self):
        widths = ProxyAssumptions(parameter_scalar_bytes=4, input_scalar_bytes=2, active_index_bytes=4)
        result = estimate_prediction("sparse_shared_index", 1, widths)
        self.assertEqual(4, result.parameter_scalar_bytes_assumed)
        self.assertEqual(2, result.input_scalar_bytes_assumed)
        self.assertEqual(4, result.active_index_bytes_assumed)
        self.assertEqual("NOT_MEASURED", result.physical_energy_joules)
        for value in (-1, 17, True, 0.5):
            with self.subTest(value=value), self.assertRaises(ValueError):
                estimate_prediction("dense_sgd", value)


if __name__ == "__main__":
    unittest.main()
