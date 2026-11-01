"""CL-8J GCP feasibility is a static NO_GO checklist, not a cloud runner."""
import json
from pathlib import Path
import unittest

BASE = Path(__file__).resolve().parents[1] / "integrations/continual_learning"


class CL8JGcpFeasibilityTests(unittest.TestCase):
    def test_design_only_never_authorizes_resources_or_models(self):
        record = json.loads((BASE / "cl8j_gcp_feasibility.json").read_text())
        self.assertEqual(record["kind"], "CL8J_GCP_FEASIBILITY_DESIGN_ONLY")
        self.assertEqual(record["decision"], "NO_GO")
        self.assertTrue(record["review_only"])
        self.assertEqual(record["selection_status"], "NOT_SELECTED")
        for key in ("account_access_confirmed", "billing_eligibility_confirmed",
                    "vm_provisioned", "project_created", "external_egress_deny_installed",
                    "controlled_egress_probe_run", "external_resource_budgets_verified",
                    "cost_cap_guaranteed", "model_execution_authorized",
                    "training_authorized", "inference_authorized"):
            self.assertIs(record[key], False)
        gates = record["design_steps"]
        self.assertEqual(len(gates), 8)
        self.assertEqual(len({gate["id"] for gate in gates}), len(gates))
        self.assertTrue(all(gate["verified"] is False for gate in gates))

    def test_upstream_no_go_remains(self):
        for path in ("cl8h_alternative_host_security_contract.json",
                     "cl8c_cpu_execution_envelope.json"):
            upstream = json.loads((BASE / path).read_text())
            self.assertEqual(upstream["decision"], "NO_GO")


if __name__ == "__main__":
    unittest.main()
