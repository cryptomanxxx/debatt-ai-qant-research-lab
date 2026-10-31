"""One-step numerical and source-level overhead checks, no dataset fitting."""
import unittest

from integrations.continual_learning.cl2_operation_accounting import estimate_step
from integrations.continual_learning.cl2_payload_proxy import estimate_payload_proxy
from integrations.continual_learning.cl2_shared_index_forward import SharedIndexSparseForwardLocalMLP
from integrations.continual_learning.cl2_sparse_forward import SparseForwardLocalMLP


class SharedIndexContracts(unittest.TestCase):
    def test_same_predictions_losses_and_post_step_parameters(self):
        for seed in (0, 101):
            for x in ((0,) * 16, (1, 0) + (0,) * 14, (1, 0) * 8, (1,) * 16):
                original = SparseForwardLocalMLP(seed=seed)
                shared = SharedIndexSparseForwardLocalMLP(seed=seed)
                self.assertEqual(original.predict(x), shared.predict(x))
                self.assertAlmostEqual(original.step(x, 1), shared.step(x, 1), places=12)
                for name in ("w1", "w2"):
                    for a, b in zip(getattr(original, name), getattr(shared, name)):
                        for lhs, rhs in zip(a, b):
                            self.assertAlmostEqual(lhs, rhs, places=12)
                for name in ("b1", "b2"):
                    for lhs, rhs in zip(getattr(original, name), getattr(shared, name)):
                        self.assertAlmostEqual(lhs, rhs, places=12)
                self.assertEqual(original.logical_counters, shared.logical_counters)

    def test_single_scan_accounting(self):
        for active in (0, 1, 8, 16):
            original = estimate_step("sparse_local_hybrid", active)
            shared = estimate_step("sparse_shared_index", active)
            self.assertEqual(original.forward_weight_terms, shared.forward_weight_terms)
            self.assertEqual(original.weight_reads, shared.weight_reads)
            self.assertEqual(original.weight_writes, shared.weight_writes)
            self.assertEqual(32, original.input_feature_checks)
            self.assertEqual(16, shared.input_feature_checks)
            old_bytes = estimate_payload_proxy("sparse_local_hybrid", active)
            new_bytes = estimate_payload_proxy("sparse_shared_index", active)
            self.assertEqual(2 * 16 + 2 * 18 * active, old_bytes.input_check_payload_bytes + old_bytes.active_index_access_payload_bytes)
            self.assertEqual(16 + 2 * 17 * active, new_bytes.input_check_payload_bytes + new_bytes.active_index_access_payload_bytes)
            self.assertLess(new_bytes.total_logical_payload_bytes, old_bytes.total_logical_payload_bytes)


if __name__ == "__main__":
    unittest.main()
