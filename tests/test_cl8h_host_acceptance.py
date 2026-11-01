"""Review-only CL-8H host-security acceptance contract; no host or model execution."""
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "integrations/continual_learning"


class CL8HHostAcceptanceTests(unittest.TestCase):
    def test_unverified_host_is_not_approved(self):
        record = json.loads((BASE / "cl8h_alternative_host_security_contract.json").read_text())
        self.assertEqual(record["kind"], "CL8H_ALTERNATIVE_HOST_SECURITY_CONTRACT")
        self.assertEqual(record["decision"], "NO_GO")
        self.assertTrue(record["review_only"])
        self.assertEqual(record["status"], "DESIGN_ONLY")
        for key in ("host_selected", "host_evidence_verified", "general_runner_implemented",
                    "model_execution_authorized", "training_authorized", "inference_authorized",
                    "independent_security_review_complete", "independent_scientific_review_complete",
                    "separate_marcus_approval"):
            self.assertIs(record[key], False)
        gates = record["required_gates"]
        self.assertEqual(len(gates), 9)
        self.assertEqual(len({gate["id"] for gate in gates}), len(gates))
        self.assertTrue(all(gate["verified"] is False and gate["proof"] for gate in gates))

    def test_upstream_no_go_not_overridden(self):
        record = json.loads((BASE / "cl8h_alternative_host_security_contract.json").read_text())
        envelope = json.loads((BASE / record["future_envelope"]).read_text())
        previous = json.loads((BASE / record["source_decision"]).read_text())
        self.assertEqual(envelope["decision"], "NO_GO")
        self.assertIs(envelope["execution_entrypoint_available"], False)
        self.assertEqual(previous["decision"], "NO_GO")


if __name__ == "__main__":
    unittest.main()
