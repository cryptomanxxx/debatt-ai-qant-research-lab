"""CL-6E static protocol checks only: never import model code or create snapshots."""
from __future__ import annotations

import json
import math
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1] / "integrations/continual_learning"


class CL6EPairedInitialStateContractTests(unittest.TestCase):
    def test_no_go_and_no_model_execution(self):
        draft = json.loads((ROOT / "cl6e_paired_initial_state_contract.json").read_text(encoding="utf-8"))
        cl6a = json.loads((ROOT / "cl6a_readiness_protocol.json").read_text(encoding="utf-8"))
        self.assertEqual(draft["decision"], "NO_GO")
        self.assertEqual(cl6a["decision"], "NO_GO")
        self.assertEqual(draft["cl6a_open_gates"], len(cl6a["open_gates"]))
        self.assertIn("causal_arm_equivalence", cl6a["open_gates"])
        for key in (
            "model_instantiation_authorized", "inference_authorized",
            "training_authorized", "data_generation_authorized",
            "state_snapshots_present", "measurement_performed",
        ):
            self.assertIs(draft[key], False, key)
        self.assertEqual(draft["future_snapshot_contract"]["comparison_policies"]["float_tolerance"], "NOT_APPROVED")

    def test_pre_update_state_fields_match_preregistered_shape(self):
        draft = json.loads((ROOT / "cl6e_paired_initial_state_contract.json").read_text(encoding="utf-8"))
        cl6a = json.loads((ROOT / "cl6a_readiness_protocol.json").read_text(encoding="utf-8"))
        contract = draft["future_snapshot_contract"]
        self.assertEqual(contract["architecture_shape"], cl6a["model"]["shape"])
        fields = contract["parameter_state"]
        self.assertEqual([field["name"] for field in fields], ["w1", "b1", "w2", "b2"])
        self.assertEqual([field["shape"] for field in fields], [[8, 16], [8], [2, 8], [2]])
        self.assertEqual(sum(math.prod(field["shape"]) for field in fields), cl6a["model"]["parameters_per_arm"])
        self.assertEqual(
            contract["paired_local_arms"],
            ["local_hybrid", "sparse_local_hybrid", "sparse_shared_index"],
        )
        self.assertEqual(contract["comparison_policies"]["unavailable_or_mismatched_state"], "NO_GO")


if __name__ == "__main__":
    unittest.main()
