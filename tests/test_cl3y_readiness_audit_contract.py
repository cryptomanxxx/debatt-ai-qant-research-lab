"""CL-3Y audit contracts, metadata only; no model execution."""
import unittest
from unittest.mock import patch

from integrations.continual_learning.cl3y_readiness_audit import (
    GATE_REVIEWS, scientific_readiness_audit,
)


class CL3YReadinessContracts(unittest.TestCase):
    def test_nine_open_gates_and_no_compute(self):
        result = scientific_readiness_audit()
        self.assertEqual("NO_GO", result["decision"])
        self.assertEqual(9, result["gate_count"])
        self.assertEqual(9, result["open_count"])
        self.assertEqual(0, result["closed_count"])
        self.assertFalse(result["training_allowed"])
        self.assertFalse(result["model_inference_allowed"])
        self.assertFalse(result["energy_measured"])
        self.assertFalse(result["external_requests_allowed"])
        self.assertEqual(9, len(set(row["gate_id"] for row in result["gates"])))
        self.assertTrue(all(row["status"] == "OPEN_NOT_INDEPENDENTLY_APPROVED"
                            and row["automatically_closable"] is False for row in result["gates"]))

    def test_no_gate_can_be_unmapped_or_silently_reclassified(self):
        with patch.dict(GATE_REVIEWS, {"source_hash_and_dataset_digest_pin": ("P1", "arbitrary", "arbitrary")}, clear=True):
            with self.assertRaises(ValueError):
                scientific_readiness_audit()

    def test_separate_human_authorization_is_still_required(self):
        result = scientific_readiness_audit()
        ids = [r["gate_id"] for r in result["gates"]]
        self.assertIn("independent_scientific_review", ids)
        self.assertIn("separate_explicit_marcus_compute_authorization", ids)
        self.assertIn("runner_source_and_artifact_schema_review", ids)


if __name__ == "__main__":
    unittest.main()
