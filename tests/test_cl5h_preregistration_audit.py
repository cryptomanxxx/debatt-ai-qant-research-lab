"""CL-5H offline preregistration audit tests; no dataset generation."""
import unittest

from integrations.continual_learning.cl5h_preregistration_audit import (
    cl5h_preregistration_audit,
)


class CL5HContractTests(unittest.TestCase):
    def test_draft_remains_no_go(self):
        report = cl5h_preregistration_audit()
        self.assertEqual(report["decision"], "NO_GO")
        self.assertFalse(report["data_generation_authorized"])
        self.assertFalse(report["experiment_training_authorized"])
        self.assertFalse(report["dataset_generation_executed"])
        self.assertFalse(report["model_training_executed"])
        self.assertEqual(report["scientific_gates_open"], 9)
        self.assertIn("rejection_attempts_by_task_and_partition",
                      report["future_result_fields"])
        self.assertIn("empirical_activity_by_task_and_partition",
                      report["future_result_fields"])


if __name__ == "__main__":
    unittest.main()
