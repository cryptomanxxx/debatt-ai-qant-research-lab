"""CL-6C preregistration invariants only; no models are imported or run."""
from __future__ import annotations

import json
import unittest
from pathlib import Path


PROTOCOL = (
    Path(__file__).resolve().parents[1]
    / "integrations/continual_learning/cl6c_numerical_parity_protocol.json"
)
CL6A = PROTOCOL.with_name("cl6a_readiness_protocol.json")


class CL6CProtocolTests(unittest.TestCase):
    def test_fail_closed_without_execution(self):
        protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
        parent = json.loads(CL6A.read_text(encoding="utf-8"))
        self.assertEqual(protocol["decision"], "NO_GO")
        for field in (
            "execution_permitted",
            "model_instantiation_authorized",
            "inference_authorized",
            "training_authorized",
            "dataset_generation_authorized",
            "physical_energy_measured",
            "tolerance_preapproved",
            "numerical_results_present",
            "scientific_equivalence_approved",
        ):
            self.assertIs(protocol[field], False, field)
        self.assertEqual(protocol["cl6a_total_open_gates"], len(parent["open_gates"]))
        self.assertEqual(parent["decision"], "NO_GO")
        self.assertIn("causal_arm_equivalence", parent["open_gates"])

    def test_exact_paired_local_contrasts(self):
        protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
        parent = json.loads(CL6A.read_text(encoding="utf-8"))
        expected = {
            (item["id"], item["left"], item["right"])
            for item in parent["comparisons"] if item["id"] != "context_learning_rule"
        }
        observed = {
            (item["id"], item["left"], item["right"])
            for item in protocol["comparisons"]
            if item["learning_rule_must_match"] is True
        }
        self.assertEqual(expected, observed)
        self.assertEqual(len(protocol["comparisons"]), 2)
        self.assertGreaterEqual(len(protocol["required_future_checks"]), 8)


if __name__ == "__main__":
    unittest.main()
