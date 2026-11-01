"""CL-3V logical-component conservation tests; never train or predict."""
import unittest

from integrations.continual_learning.cl2_payload_proxy import ProxyAssumptions
from integrations.continual_learning.cl3v_cost_components import ARMS, PHASES, component_report, decompose


class CL3VContracts(unittest.TestCase):
    def test_all_activity_arm_phase_components_reconcile(self):
        report = component_report()
        self.assertEqual(136, len(report["rows"]))
        self.assertFalse(report["training_executed"])
        self.assertFalse(report["inference_executed"])
        self.assertFalse(report["energy_measured"])
        for row in report["rows"]:
            self.assertEqual(row["total_bytes"], sum(row[k] for k in ("parameter_bytes", "input_bytes", "index_bytes")))
            self.assertGreater(row["parameter_bytes"], 0)
            self.assertGreaterEqual(row["index_bytes"], 0)

    def test_reference_values_and_zero_index_for_dense(self):
        for phase in PHASES:
            for active in range(17):
                self.assertEqual(0, decompose("dense_sgd", phase, active)["index_bytes"])
        sparse = decompose("sparse_local_hybrid", "inference", 2)
        self.assertEqual({"parameter_bytes": 288, "input_bytes": 32, "index_bytes": 84,
                          "total_bytes": 404}, {k: sparse[k] for k in ("parameter_bytes", "input_bytes", "index_bytes", "total_bytes")})
        self.assertEqual(4080, decompose("dense_sgd", "training", 2)["total_bytes"])

    def test_width_sensitivity_and_invalid_values(self):
        self.assertNotEqual(
            decompose("sparse_local_hybrid", "training", 2, ProxyAssumptions(8, 1, 2))["index_bytes"],
            decompose("sparse_local_hybrid", "training", 2, ProxyAssumptions(8, 1, 4))["index_bytes"],
        )
        for invalid in (-1, 17, True):
            with self.assertRaises(ValueError):
                decompose("dense_sgd", "inference", invalid)
        with self.assertRaises(ValueError):
            decompose("other", "training", 2)
        with self.assertRaises(ValueError):
            decompose("dense_sgd", "other", 2)


if __name__ == "__main__":
    unittest.main()
