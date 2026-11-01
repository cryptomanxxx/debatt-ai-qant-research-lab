"""CL-5I AST-only test. It does not execute the generator."""
import unittest

from integrations.continual_learning.cl5i_static_generator_probe_audit import (
    static_generator_probe_audit,
)


class StaticProbeAuditTests(unittest.TestCase):
    def test_reviewed_loop_and_no_go_contract(self):
        report = static_generator_probe_audit()
        self.assertEqual(report["decision"], "NO_GO")
        self.assertEqual(report["attempt_loop_cap_per_partition"], 1_000_000)
        self.assertEqual(len(report["reviewed_statement_order"]), 7)
        self.assertFalse(report["instrumentation_implemented"])
        self.assertFalse(report["generator_modified"])
        self.assertFalse(report["dataset_generation_executed"])
        self.assertFalse(report["model_training_executed"])
        self.assertEqual(report["scientific_gates_open"], 9)


if __name__ == "__main__":
    unittest.main()
