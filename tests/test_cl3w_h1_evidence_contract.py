"""CL-3W no-compute evidence gate and scientific-claim contracts."""
import unittest

from integrations.continual_learning.cl3w_h1_evidence import h1_evidence_review


class CL3WEvidenceContracts(unittest.TestCase):
    def test_h1_static_claims_are_bounded_and_reconciled(self):
        result = h1_evidence_review()
        self.assertEqual("CL3W_H1_STATIC_EVIDENCE_REVIEW_NOT_EXPERIMENT", result["kind"])
        self.assertEqual(["H1-S1", "H1-S2", "H1-S3"], [x["id"] for x in result["claims"]])
        one, two, three = result["claims"]
        self.assertEqual((1376, 404), (one["dense_bytes"], one["sparse_bytes"]))
        self.assertEqual((1376, 1224), (two["dense_bytes"], two["sparse_bytes"]))
        self.assertEqual(14, two["first_not_cheaper_activity"])
        self.assertEqual(9, three["scenario_count"])
        self.assertTrue(three["nonreference_shapes_are_extrapolations"])

    def test_unsupported_results_never_promoted_to_verified(self):
        result = h1_evidence_review()
        self.assertFalse(result["training_performed"])
        self.assertFalse(result["inference_performed"])
        self.assertFalse(result["energy_measured"])
        self.assertIn("NO_GO", result["decision"])
        for claim in result["unsupported_claims"].values():
            self.assertIn(claim, ("NOT_MEASURED", "NOT_TESTED"))
        for claim in result["claims"]:
            self.assertNotIn("VERIFIED_HARDWARE", claim["status"])


if __name__ == "__main__":
    unittest.main()
