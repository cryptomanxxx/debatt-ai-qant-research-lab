"""CL-11 integration tests using only hand-authored fixture rows, NO MODEL."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cl11_integrated_noexec_handoff as cl11
from cl10c_artifact_writer import write_artifact_bundle
from test_cl10d_h1_coverage_audit import fixtures


class TestCL11IntegratedHandoff(unittest.TestCase):
    def test_review_only_preflight_never_grants_execution(self):
        gate = cl11.integrated_review_only_preflight()
        self.assertEqual(gate["decision"], "NO_GO")
        self.assertEqual(gate["planned_pairs"], 20)
        self.assertEqual(gate["planned_training_updates_max"], 10240)
        self.assertEqual(gate["observations_recorded"], 0)
        self.assertEqual(gate["models_constructed"], 0)
        self.assertEqual(gate["datasets_generated"], 0)
        self.assertFalse(gate["network_isolation_approved"])
        self.assertFalse(gate["training_authorized"])
        self.assertFalse(gate["inference_authorized"])
        self.assertEqual(len(gate["blockers"]), 9)

    def test_supplied_fixture_ledger_is_never_evidence_or_go(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle = write_artifact_bundle(Path(tmp), "cl11-fixture", **fixtures())
            report = cl11.assess_supplied_artifact_handoff(bundle)
            self.assertEqual(report["decision"], "NO_GO")
            self.assertEqual(report["heldout_records"], 400)
            self.assertEqual(report["training_attempts"], 80)
            self.assertEqual(report["source_event_records"], 0)
            self.assertFalse(report["source_event_matrix_complete"])
            self.assertFalse(report["source_event_authenticity_verified"])
            self.assertFalse(report["scientific_signoff"])
            self.assertFalse(report["training_authorized"])
            with (bundle / "heldout_accuracy.jsonl").open("ab") as stream:
                stream.write(b" ")
            with self.assertRaises(ValueError):
                cl11.assess_supplied_artifact_handoff(bundle)

    def test_changed_execution_authorization_fails_closed(self):
        proposal = json.loads(cl11.PROPOSAL.read_text(encoding="utf-8"))
        proposal["training_authorized"] = True
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "altered.json"
            path.write_text(json.dumps(proposal), encoding="utf-8")
            with patch.object(cl11, "PROPOSAL", path):
                with self.assertRaises(ValueError):
                    cl11.integrated_review_only_preflight()

    def test_changed_network_claim_fails_closed(self):
        proposal = json.loads(cl11.PROPOSAL.read_text(encoding="utf-8"))
        proposal["execution_budget"]["network_egress_policy"] = "APPROVED"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "altered.json"
            path.write_text(json.dumps(proposal), encoding="utf-8")
            with patch.object(cl11, "PROPOSAL", path):
                with self.assertRaises(ValueError):
                    cl11.integrated_review_only_preflight()


if __name__ == "__main__":
    unittest.main()
