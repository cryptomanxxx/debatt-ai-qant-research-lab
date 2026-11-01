"""CL-9G static reconciliation regressions. No models, inference or datasets."""
import copy
import unittest
from unittest.mock import patch

from integrations.continual_learning import cl9g_protocol_consistency as audit


class CL9GProtocolConsistencyTests(unittest.TestCase):
    def test_live_contracts_are_consistent_but_no_go(self):
        result = audit.reconcile()
        self.assertEqual(result, {
            "decision": "NO_GO", "seeds": 5, "arms": 4,
            "causal_contrasts": 2, "open_gates": 9,
            "open_scientific_findings": 6, "model_runs": 0,
            "physical_energy": "NOT_MEASURED",
        })

    def mutate(self, name, change):
        original = audit._read

        def altered(path):
            data = copy.deepcopy(original(path))
            if path == name:
                change(data)
            return data

        return patch.object(audit, "_read", side_effect=altered)

    def test_seed_mismatch_fails_closed(self):
        with self.mutate("cl9b_h1_acceptance_contract.json",
                         lambda data: data["descriptive_seeds"].pop()):
            with self.assertRaisesRegex(ValueError, "seeds differ"):
                audit.reconcile()

    def test_swapped_causal_arm_fails_closed(self):
        with self.mutate("cl7d_causal_pairing_invariants.json",
                         lambda data: data["causal_pairs"][0].update(right="dense_sgd")):
            with self.assertRaisesRegex(ValueError, "contrasts differ"):
                audit.reconcile()

    def test_training_approval_field_must_be_exact_false(self):
        for value in (None, "false", True, 0):
            with self.subTest(value=value):
                with self.mutate("cl9c_measurement_protocol.json",
                                 lambda data: data.update(model_training_authorized=value)):
                    with self.assertRaisesRegex(ValueError, "must be false"):
                        audit.reconcile()

    def test_open_gate_cannot_be_silently_closed(self):
        with self.mutate("cl9a_minimal_h1_readiness.json",
                         lambda data: data["minimum_gates"][0].update(status="CLOSED")):
            with self.assertRaisesRegex(ValueError, "prematurely closed"):
                audit.reconcile()

    def test_heldout_calibration_leakage_fails_closed(self):
        with self.mutate("cl7c_train_only_calibration_protocol.json",
                         lambda data: data["calibration_split"].update(
                             heldout_test_labels_eligible=True)):
            with self.assertRaisesRegex(ValueError, "heldout labels"):
                audit.reconcile()

    def test_energy_claim_cannot_replace_not_measured(self):
        with self.mutate("cl9c_measurement_protocol.json",
                         lambda data: data.update(physical_energy="MEASURED")):
            with self.assertRaisesRegex(ValueError, "energy status"):
                audit.reconcile()


if __name__ == "__main__":
    unittest.main()
