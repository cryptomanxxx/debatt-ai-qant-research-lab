"""CL-8G deterministic review-only host decision contracts; no subprocesses."""
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "integrations/continual_learning/cl8g_host_isolation_decision.json"


class CL8GHostDecisionTests(unittest.TestCase):
    def test_no_go_cannot_be_mistaken_for_approval(self):
        record = json.loads(RECORD.read_text(encoding="utf-8"))
        self.assertEqual(record["kind"], "CL8G_HOST_ISOLATION_DECISION")
        self.assertEqual(record["decision"], "NO_GO")
        self.assertTrue(record["review_only"])
        for field in ("model_execution_authorized", "training_authorized", "inference_authorized"):
            self.assertIs(record[field], False)
        self.assertIs(record["gate_status"]["independent_scientific_signoff"], False)
        self.assertIs(record["gate_status"]["separate_marcus_training_authorization"], False)

    def test_observations_are_distinct_from_unverified_candidates(self):
        record = json.loads(RECORD.read_text(encoding="utf-8"))
        self.assertEqual([e["run_id"] for e in record["evidence"]], [41343095078, 41343841231])
        self.assertIn("OS_PERMISSION_DENIED", record["evidence"][1]["observation"])
        self.assertTrue(all(x["status"] == "UNVERIFIED" for x in record["host_options"]))
        self.assertEqual(record["gate_status"]["verified_network_denial"], "NOT_DEMONSTRATED")


if __name__ == "__main__":
    unittest.main()
