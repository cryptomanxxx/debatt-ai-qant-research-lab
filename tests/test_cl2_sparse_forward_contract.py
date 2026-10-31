"""Sparse-vs-dense numerical equivalence contracts; not a dataset run."""
import unittest

from integrations.continual_learning.cl2_local_mlp import LocalHebbianMLP
from integrations.continual_learning.cl2_operation_accounting import estimate_step
from integrations.continual_learning.cl2_sparse_forward import SparseForwardLocalMLP


class SparseForwardContracts(unittest.TestCase):
    def test_equivalent_forward_on_multiple_activity_patterns(self):
        for seed in (0, 101):
            dense = LocalHebbianMLP(seed=seed)
            sparse = SparseForwardLocalMLP(seed=seed)
            patterns = (
                (0,) * 16,
                (1,) * 16,
                (1, 0) + (0,) * 14,
                (1, 0) * 8,
                (0, 1) * 8,
                (1, 1, 0, 0) * 4,
            )
            for x in patterns:
                h1, logits1 = dense._forward(x)
                h2, logits2 = sparse._forward(x)
                for a, b in zip(h1 + logits1, h2 + logits2):
                    self.assertAlmostEqual(a, b, places=12)
                self.assertEqual(dense.predict(x), sparse.predict(x))

    def test_one_matched_local_update_and_counter(self):
        for x in ((0,) * 16, (1, 0) + (0,) * 14, (1,) * 16):
            dense = LocalHebbianMLP(seed=101)
            sparse = SparseForwardLocalMLP(seed=101)
            loss_a = dense.step(x, 1)
            loss_b = sparse.step(x, 1)
            self.assertAlmostEqual(loss_a, loss_b, places=12)
            for name in ("w1", "w2"):
                for old_row, new_row in zip(getattr(dense, name), getattr(sparse, name)):
                    for a, b in zip(old_row, new_row):
                        self.assertAlmostEqual(a, b, places=12)
            for name in ("b1", "b2"):
                for a, b in zip(getattr(dense, name), getattr(sparse, name)):
                    self.assertAlmostEqual(a, b, places=12)
            active = sum(x)
            profile = estimate_step("sparse_local_hybrid", active)
            self.assertEqual(profile.weight_reads, sparse.logical_counters.synaptic_reads)
            self.assertEqual(profile.weight_writes, sparse.logical_counters.weight_writes)

    def test_static_scope_shows_sparse_forward_and_scan_overhead(self):
        for active in (0, 1, 8, 16):
            dense = estimate_step("local_hybrid", active)
            sparse = estimate_step("sparse_local_hybrid", active)
            self.assertEqual(16 + 8 * active, sparse.forward_weight_terms)
            self.assertEqual(32 + 16 * active, sparse.weight_reads)
            self.assertEqual(dense.weight_writes, sparse.weight_writes)
            self.assertEqual(32, sparse.input_feature_checks)
        self.assertEqual(
            estimate_step("local_hybrid", 16).weight_reads,
            estimate_step("sparse_local_hybrid", 16).weight_reads,
        )


if __name__ == "__main__":
    unittest.main()
