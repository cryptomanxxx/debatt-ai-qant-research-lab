"""CL-6F protocol-only consistency checks; no model implementation imports."""
from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1] / "integrations/continual_learning"


def _read(name: str) -> dict:
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


class CL6FStepwiseStateProtocolTests(unittest.TestCase):
    def test_never_authorizes_execution_or_closes_parent_gates(self):
        protocol = _read("cl6f_stepwise_state_diff_protocol.json")
        parent = _read("cl6a_readiness_protocol.json")
        self.assertEqual(protocol["decision"], "NO_GO")
        self.assertEqual(parent["decision"], "NO_GO")
        self.assertEqual(protocol["cl6a_open_gates"], len(parent["open_gates"]))
        self.assertEqual(protocol["causal_arm_equivalence_gate"], "OPEN")
        self.assertTrue(protocol["separate_marcus_training_approval_required"])
        for key in (
            "model_instantiation_authorized", "model_forward_authorized",
            "training_authorized", "dataset_generation_authorized",
            "run_executed", "result_records_present", "energy_measured",
        ):
            self.assertIs(protocol[key], False, key)

    def test_paired_arms_and_state_contract_match_prior_protocols(self):
        protocol = _read("cl6f_stepwise_state_diff_protocol.json")
        parent = _read("cl6a_readiness_protocol.json")
        initial = _read("cl6e_paired_initial_state_contract.json")
        expected = [
            {k: item[k] for k in ("id", "left", "right")}
            for item in parent["comparisons"]
            if item["id"] != "context_learning_rule"
        ]
        self.assertEqual(protocol["paired_comparisons"], expected)
        names = [item["name"] for item in initial["future_snapshot_contract"]["parameter_state"]]
        record = protocol["future_record"]
        for field in names:
            self.assertIn(field, record["pre_update_state_fields"])
            self.assertIn(field, record["post_update_state_fields"])
        nonparam = initial["future_snapshot_contract"]["common_nonparameter_state"]
        local_only = initial["future_snapshot_contract"]["local_only_state"]
        for field in nonparam + local_only:
            self.assertIn(field, record["pre_update_state_fields"])
            self.assertIn(field, record["post_update_state_fields"])
        self.assertIn("learning_rate", record["pre_update_state_fields"])
        self.assertIn("local_rate", record["post_update_state_fields"])
        self.assertEqual(record["compare_at"], ["pre_update", "post_each_update"])
        self.assertEqual(record["tolerance_status"], "UNAPPROVED")
        self.assertEqual(record["missing_data_result"], "NO_GO")
        self.assertEqual(record["mismatch_result"], "NO_GO")
        self.assertEqual(record["hardware_energy_result"], "NOT_MEASURED")
        self.assertTrue(record["require_equal_sample_and_label_order"])


if __name__ == "__main__":
    unittest.main()
