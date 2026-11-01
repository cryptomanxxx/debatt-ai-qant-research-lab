"""CL-8I preliminary low-cost host screening: review-only tests, no compute."""
import json
from pathlib import Path
import unittest

BASE = Path(__file__).resolve().parents[1] / "integrations/continual_learning"


class CL8ILowCostHostScreenTests(unittest.TestCase):
    def test_no_go_and_no_provisioning(self):
        record = json.loads((BASE / "cl8i_low_cost_host_screen.json").read_text())
        self.assertEqual(record["kind"], "CL8I_COST_CONSCIOUS_HOST_SCREEN")
        self.assertTrue(record["review_only"])
        self.assertEqual(record["decision"], "NO_GO")
        self.assertIsNone(record["selected_host"])
        self.assertEqual(record["provisioned_hosts"], 0)
        self.assertIs(record["network_policy_verified"], False)
        for key in ("model_execution_authorized", "training_authorized", "inference_authorized"):
            self.assertIs(record[key], False)
        self.assertEqual(len(record["options"]), 4)
        self.assertEqual(len({x["id"] for x in record["options"]}), 4)
        self.assertTrue(all(x["host_acceptance"] == "NO_GO" for x in record["options"]))

    def test_no_approved_host_hidden_in_rankings(self):
        record = json.loads((BASE / "cl8i_low_cost_host_screen.json").read_text())
        ids = {x["id"] for x in record["options"]}
        self.assertTrue(set(record["ranking_for_next_no_model_security_study"]) <= ids)
        self.assertTrue(record["ranking_is_preliminary"])
        host = json.loads((BASE / "cl8h_alternative_host_security_contract.json").read_text())
        self.assertEqual(host["decision"], "NO_GO")
        self.assertTrue(all(gate["verified"] is False for gate in host["required_gates"]))


if __name__ == "__main__":
    unittest.main()
