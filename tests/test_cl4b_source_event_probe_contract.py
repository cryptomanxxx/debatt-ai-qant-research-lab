"""CL-4B deterministic loop-probe tests: no model instances or training."""
import unittest

from integrations.continual_learning.cl4b_source_event_probe import (
    SOURCE_SCOPE, source_loop_probe,
)
from integrations.continual_learning.cl4a_h1_measurement_contract import COUNTERS


class CL4BSourceEventContracts(unittest.TestCase):
    def test_two_active_bits_expose_actual_loop_event_difference(self):
        x = (1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0)
        dense = source_loop_probe(x, sparse=False)
        sparse = source_loop_probe(x, sparse=True)
        self.assertEqual(144, dense["counters"]["forward_weight_terms"])
        self.assertEqual(32, sparse["counters"]["forward_weight_terms"])
        self.assertEqual(128, dense["counters"]["input_visits"])
        self.assertEqual(16, sparse["counters"]["input_visits"])
        self.assertEqual(16, sparse["counters"]["index_reads"])
        self.assertEqual(2, sparse["counters"]["index_writes"])
        self.assertEqual(0, dense["counters"]["index_reads"])
        self.assertEqual(8, sparse["counters"]["nonlinearity_counts"])
        self.assertEqual(0, sparse["counters"]["parameter_writes"])

    def test_activity_sweep_and_dense_invariance(self):
        dense_terms=set()
        for activity_count in range(17):
            x = (1,) * activity_count + (0,) * (16 - activity_count)
            dense = source_loop_probe(x, sparse=False)
            sparse = source_loop_probe(x, sparse=True)
            dense_terms.add(dense["counters"]["forward_weight_terms"])
            self.assertEqual(16 + 8 * activity_count,
                             sparse["counters"]["forward_weight_terms"])
            self.assertEqual(activity_count, sparse["active_inputs"])
            self.assertEqual(set(COUNTERS), set(sparse["counters"]))
        self.assertEqual({144}, dense_terms)

    def test_no_model_training_energy_or_hardware_claim(self):
        x = (0,) * 16
        for sparse in (False, True):
            result = source_loop_probe(x, sparse=sparse)
            self.assertEqual(SOURCE_SCOPE, result["scope"])
            for field in ("model_executed", "training_executed", "inference_executed"):
                self.assertFalse(result[field])
            for field in ("energy_joules", "cpu_instructions", "hardware_memory_bytes"):
                self.assertIsNone(result[field])

    def test_rejects_nonbinary_shapes_and_ambiguous_sparse_flag(self):
        for x in ((0,) * 15, (0,) * 17, (0,) * 15 + (2,), (False,) * 16, [0] * 16):
            with self.subTest(x=str(x)[:30]), self.assertRaises(ValueError):
                source_loop_probe(x, sparse=True)
        with self.assertRaises(ValueError):
            source_loop_probe((0,) * 16, sparse=1)


if __name__ == "__main__":
    unittest.main()
