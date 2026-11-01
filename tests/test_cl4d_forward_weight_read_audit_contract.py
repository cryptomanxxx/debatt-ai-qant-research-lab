"""CL-4D tiny deterministic unit forwards, never any training or dataset run."""
import math
import unittest

from integrations.continual_learning.cl4b_source_event_probe import source_loop_probe
from integrations.continual_learning.cl4d_forward_weight_read_audit import audit_single_forward


class CL4DForwardReadContracts(unittest.TestCase):
    def test_two_active_bits_match_isolated_probe_and_numerical_outputs(self):
        features = (1,) + (0,) * 7 + (1,) + (0,) * 7
        dense = audit_single_forward("local_hybrid", features)
        sparse = audit_single_forward("sparse_local_hybrid", features)
        self.assertEqual((128, 16), (dense.w1_reads, dense.w2_reads))
        self.assertEqual((16, 16), (sparse.w1_reads, sparse.w2_reads))
        self.assertEqual(dense.w1_reads + dense.w2_reads,
                         source_loop_probe(features, sparse=False)["counters"]["forward_weight_terms"])
        self.assertEqual(sparse.w1_reads + sparse.w2_reads,
                         source_loop_probe(features, sparse=True)["counters"]["forward_weight_terms"])
        self.assertTrue(all(math.isclose(a, b, rel_tol=0, abs_tol=1e-12)
                            for a, b in zip(dense.logits, sparse.logits)))
        self.assertFalse(dense.training_executed)
        self.assertFalse(sparse.task_inference_executed)
        self.assertIsNone(sparse.energy_joules)

    def test_full_activity_sweep_records_model_weight_row_getitem_events(self):
        for active in range(17):
            features = (1,) * active + (0,) * (16 - active)
            dense = audit_single_forward("local_hybrid", features)
            sparse = audit_single_forward("sparse_local_hybrid", features)
            with self.subTest(active=active):
                self.assertEqual(128, dense.w1_reads)
                self.assertEqual(8 * active, sparse.w1_reads)
                self.assertEqual(16, dense.w2_reads)
                self.assertEqual(16, sparse.w2_reads)
                self.assertEqual(dense.active_inputs, sparse.active_inputs)
                self.assertTrue(all(math.isclose(a, b, rel_tol=0, abs_tol=1e-12)
                                    for a, b in zip(dense.logits, sparse.logits)))

    def test_rejects_unapproved_inputs_without_training(self):
        for bad in ((0,) * 15, (0,) * 15 + (2,), (False,) * 16, [0] * 16):
            with self.subTest(value=str(bad)[:30]), self.assertRaises(ValueError):
                audit_single_forward("local_hybrid", bad)
        with self.assertRaises(ValueError):
            audit_single_forward("dense_sgd", (0,) * 16)
        with self.assertRaises(ValueError):
            audit_single_forward("local_hybrid", (0,) * 16, seed=True)


if __name__ == "__main__":
    unittest.main()
