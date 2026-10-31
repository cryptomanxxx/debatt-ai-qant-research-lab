"""Pure calculation tests: no workload run, physical profiling or energy model."""
import unittest

from integrations.continual_learning.cl2_payload_proxy import (
    ProxyAssumptions,
    estimate_payload_proxy,
    sensitivity_table,
)


class PayloadProxyContracts(unittest.TestCase):
    def test_single_active_example_reproducible_payload(self):
        dense = estimate_payload_proxy("dense_sgd", 1)
        local = estimate_payload_proxy("local_hybrid", 1)
        sparse = estimate_payload_proxy("sparse_local_hybrid", 1)
        self.assertEqual(4080, dense.total_logical_payload_bytes)
        self.assertEqual(1938, local.total_logical_payload_bytes)
        self.assertEqual(884, sparse.total_logical_payload_bytes)
        self.assertEqual(36, sparse.active_index_access_payload_bytes)
        self.assertEqual(18, local.active_index_access_payload_bytes)

    def test_full_activity_index_overhead_can_dominate(self):
        dense = estimate_payload_proxy("dense_sgd", 16)
        local = estimate_payload_proxy("local_hybrid", 16)
        sparse = estimate_payload_proxy("sparse_local_hybrid", 16)
        self.assertEqual(4080, dense.total_logical_payload_bytes)
        self.assertEqual(4128, local.total_logical_payload_bytes)
        self.assertEqual(4304, sparse.total_logical_payload_bytes)
        self.assertGreater(sparse.total_logical_payload_bytes, dense.total_logical_payload_bytes)

    def test_width_sensitivity_and_validation(self):
        table = sensitivity_table(1)
        self.assertEqual(9, len(table))
        self.assertEqual(3, len({x.parameter_access_payload_bytes for x in table if x.arm == "dense_sgd"}))
        with self.assertRaises(ValueError):
            ProxyAssumptions(active_index_bytes=0)
        with self.assertRaises(ValueError):
            ProxyAssumptions(parameter_scalar_bytes=True)
        with self.assertRaises(ValueError):
            estimate_payload_proxy("dense_sgd", -1)


if __name__ == "__main__":
    unittest.main()
