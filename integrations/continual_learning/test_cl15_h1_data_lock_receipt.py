"""CL-15 receipt-only checks; no data access or model calls."""
import json
import unittest
from pathlib import Path


class TestCL15Receipt(unittest.TestCase):
    def test_exact_review_only_receipt(self):
        path = Path(__file__).with_name("cl15_h1_data_lock_candidate.json")
        value = json.loads(path.read_text(encoding="ascii"))
        self.assertEqual(value["decision"], "NO_GO")
        self.assertFalse(value["scientific_data_freeze_approved"])
        self.assertFalse(value["records_committed_to_repository"])
        self.assertFalse(value["training_authorized"])
        self.assertFalse(value["inference_authorized"])
        self.assertFalse(value["dataset_regeneration_authorized"])
        self.assertEqual(value["physical_energy"], "NOT_MEASURED")
        self.assertEqual(value["records_total"], 3840)
        self.assertEqual(value["train_records"] + value["heldout_records"], 3840)
        self.assertEqual(set(value["seed_dataset_sha256"]), {"101", "102", "103", "104", "105"})
        for digest in value["seed_dataset_sha256"].values():
            self.assertEqual(len(digest), 64)
            self.assertTrue(all(c in "0123456789abcdef" for c in digest))
        self.assertEqual(value["source_generation"]["run_id"], 41432369864)
        self.assertEqual(value["independent_audit"]["run_id"], 41433518948)


if __name__ == "__main__":
    unittest.main()
