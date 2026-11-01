"""CL-5G NO_GO evidence boundary contract tests: no sample or model compute."""
import unittest

from integrations.continual_learning.cl5g_evidence_boundary_audit import (
    cl5g_evidence_boundary_audit,
)


class EvidenceSynthesisTests(unittest.TestCase):
    def test_no_go_and_all_six_sources(self):
        report = cl5g_evidence_boundary_audit()
        self.assertEqual(report["decision"], "NO_GO")
        self.assertEqual([row["stage"] for row in report["evidence"]],
                         ["CL5A", "CL5B", "CL5C", "CL5D", "CL5E", "CL5F"])
        self.assertEqual(report["scientific_gates_open"], 9)
        self.assertEqual(report["scientific_gates_closed"], 0)
        self.assertTrue(all(row["closes_scientific_gate"] is False
                            for row in report["evidence"]))
        self.assertFalse(report["dataset_generation_executed"])
        self.assertFalse(report["model_training_executed"])
        self.assertFalse(report["measured_energy"])
        self.assertFalse(report["scientific_approval_granted"])


if __name__ == "__main__":
    unittest.main()
