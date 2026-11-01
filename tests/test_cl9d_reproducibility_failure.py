"""CL-9D: deterministic contract checks with zero model execution."""
import json
from pathlib import Path
import unittest

BASE = Path(__file__).resolve().parents[1] / "integrations/continual_learning"


class CL9DReproducibilityTests(unittest.TestCase):
    def test_append_only_failure_contract(self):
        record = json.loads((BASE / "cl9d_reproducibility_failure_policy.json").read_text())
        self.assertEqual(record["kind"], "CL9D_REPRODUCIBILITY_FAILURE_POLICY")
        self.assertEqual(record["decision"], "NO_GO")
        self.assertEqual(record["results_recorded"], 0)
        self.assertTrue(record["review_only"])
        for key in ("artifact_writer_implemented", "model_execution_authorized",
                    "training_authorized", "inference_authorized",
                    "new_data_generation_authorized", "hardware_energy_measured"):
            self.assertIs(record[key], False)
        self.assertEqual(set(record["outcome_statuses"]),
                         {"COMPLETE", "FAILED", "INCOMPLETE"})
        self.assertIn("failure_reason", record["future_seed_attempt_fields"])
        self.assertTrue(record["retry_policy"]["attempts_are_append_only"])
        self.assertFalse(record["retry_policy"]["automatic_training_retries"])
        self.assertFalse(record["reporting_policy"]["posthoc_exclusions"])
        self.assertTrue(record["reporting_policy"]["missing_runs_included_in_denominator"])

    def test_consistent_with_scientific_reporting(self):
        record = json.loads((BASE / "cl9d_reproducibility_failure_policy.json").read_text())
        measurement = json.loads((BASE / record["upstream_measurement"]).read_text())
        acceptance = json.loads((BASE / record["upstream_acceptance"]).read_text())
        self.assertEqual(set(record["outcome_statuses"]),
                         set(measurement["future_record_schema"]["allowed_run_status"]))
        self.assertEqual(record["reporting_policy"]["descriptive_n_seeds"],
                         len(acceptance["descriptive_seeds"]))
        self.assertEqual(measurement["decision"], "NO_GO")
        self.assertEqual(acceptance["decision"], "NO_GO")
        self.assertTrue(record["artifact_policy"]["required_before_execution"])
        self.assertFalse(record["artifact_policy"]["immutable_writer_ready"])


if __name__ == "__main__":
    unittest.main()
