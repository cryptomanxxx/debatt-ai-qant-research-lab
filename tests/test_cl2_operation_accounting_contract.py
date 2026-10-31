"""Equal-scope operation accounting contracts: no research training."""
import unittest

from integrations.continual_learning.cl2_local_mlp import LocalHebbianMLP
from integrations.continual_learning.cl2_operation_accounting import estimate_step


class CL2OperationAccountingTests(unittest.TestCase):
    def test_dense_forward_is_identical_in_both_arms(self):
        for active in (0, 1, 8, 16):
            dense = estimate_step("dense_sgd", active)
            local = estimate_step("local_hybrid", active)
            self.assertEqual(144, dense.forward_weight_terms)
            self.assertEqual(dense.forward_weight_terms, local.forward_weight_terms)
            self.assertEqual(304, dense.weight_reads)
            self.assertEqual(144, dense.weight_writes)
            self.assertEqual(160 + 8 * active, local.weight_reads)
            self.assertEqual(16 + 8 * active, local.weight_writes)
            self.assertEqual(20, local.bias_reads)
            self.assertEqual(10, dense.bias_writes)
            self.assertEqual(256, dense.input_feature_checks)
            self.assertEqual(144, local.input_feature_checks)

    def test_local_runtime_counter_matches_static_scope(self):
        model = LocalHebbianMLP(seed=101)
        x = (0, 1) + (0,) * 14
        model.step(x, 0)  # isolated unit update, not dataset training
        profile = estimate_step("local_hybrid", 1)
        self.assertEqual(1, model.logical_counters.events)
        self.assertEqual(profile.weight_reads, model.logical_counters.synaptic_reads)
        self.assertEqual(profile.weight_writes, model.logical_counters.weight_writes)

    def test_invalid_scope_arguments_rejected(self):
        for value in (-1, 17, True, 1.5):
            with self.subTest(value=value), self.assertRaises(ValueError):
                estimate_step("dense_sgd", value)
        with self.assertRaises(ValueError):
            estimate_step("unknown", 1)


if __name__ == "__main__":
    unittest.main()
