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

    def test_missing_or_nonboolean_authorization_fails(self):
        for value in (None, "false", True, 0):
            with self.subTest(value=value):
                with self.mutate("cl9d_reproducibility_failure_policy.json",
                                 lambda d: d.update(model_execution_authorized=value)):
                    with self.assertRaisesRegex(ValueError, "must be false"):
                        audit.reconcile()
        with self.mutate("cl7a_scientific_design_critique.json",
                         lambda d: d.update(training_authorized=True)):
            with self.assertRaisesRegex(ValueError, "must be false"):
                audit.reconcile()

    def test_checkpoint_and_measurement_heldout_leakage_fails(self):
        with self.mutate("cl7c_train_only_calibration_protocol.json",
                         lambda d: d["selection_fences"].update(heldout_checkpoint_selection=True)):
            with self.assertRaisesRegex(ValueError, "selection fence"):
                audit.reconcile()
        with self.mutate("cl9c_measurement_protocol.json",
                         lambda d: d["paired_design"].update(heldout_selection="ALLOWED")):
            with self.assertRaisesRegex(ValueError, "heldout selection"):
                audit.reconcile()

    def test_nested_energy_claims_fail(self):
        with self.mutate("cl9b_h1_acceptance_contract.json",
                         lambda d: d["work_metrics"].update(physical_energy="MEASURED")):
            with self.assertRaisesRegex(ValueError, "nested energy"):
                audit.reconcile()
        with self.mutate("cl9c_measurement_protocol.json",
                         lambda d: d["future_record_schema"]["energy_field"].update(
                             joules_field_allowed=True)):
            with self.assertRaisesRegex(ValueError, "nested energy"):
                audit.reconcile()

    def test_arm_properties_are_part_of_causal_contract(self):
        with self.mutate("cl6a_readiness_protocol.json",
                         lambda d: d["arms"][2].update(learning="global_sgd")):
            with self.assertRaisesRegex(ValueError, "arm learning"):
                audit.reconcile()
        with self.mutate("cl6a_readiness_protocol.json",
                         lambda d: d["arms"][3].pop("active_index_reuse")):
            with self.assertRaisesRegex(ValueError, "arm learning"):
                audit.reconcile()

    def test_coordinated_quality_threshold_change_fails(self):
        original = audit._read
        def changed(name):
            d = copy.deepcopy(original(name))
            if name == "cl6a_readiness_protocol.json":
                d["preregistered_quality"].update(
                    final_mean_accuracy_floor=0.0,
                    paired_accuracy_difference_max_abs=1.0)
            if name == "cl9b_h1_acceptance_contract.json":
                d["quality_gate"].update(
                    final_mean_accuracy_floor_per_seed_and_arm=0.0,
                    maximum_absolute_final_mean_paired_accuracy_gap_per_seed=1.0)
            return d
        with patch.object(audit, "_read", side_effect=changed):
            with self.assertRaisesRegex(ValueError, "frozen quality"):
                audit.reconcile()

    def test_result_flags_fail_closed(self):
        with self.mutate("cl6a_readiness_protocol.json",
                         lambda d: d.update(new_results_present=True)):
            with self.assertRaisesRegex(ValueError, "result/review/energy"):
                audit.reconcile()
        with self.mutate("cl7d_causal_pairing_invariants.json",
                         lambda d: d.pop("new_results_present")):
            with self.assertRaisesRegex(ValueError, "result/review/energy"):
                audit.reconcile()

    def test_failed_seed_exclusion_fails_closed(self):
        with self.mutate("cl9d_reproducibility_failure_policy.json",
                         lambda d: d["retry_policy"].update(failed_seed_removal=True)):
            with self.assertRaisesRegex(ValueError, "retry policy"):
                audit.reconcile()
        with self.mutate("cl9d_reproducibility_failure_policy.json",
                         lambda d: d["reporting_policy"].update(
                             missing_runs_included_in_denominator=False)):
            with self.assertRaisesRegex(ValueError, "failure reporting"):
                audit.reconcile()
        with self.mutate("cl9d_reproducibility_failure_policy.json",
                         lambda d: d["reporting_policy"].update(
                             quality_matched_savings_on_missing_or_failed_pair=True)):
            with self.assertRaisesRegex(ValueError, "failure reporting"):
                audit.reconcile()

    def test_grouped_calibration_split_cannot_be_relaxed(self):
        with self.mutate("cl7c_train_only_calibration_protocol.json",
                         lambda d: d["calibration_split"].update(
                             group_identical_vectors_together=False, split_unit="record")):
            with self.assertRaisesRegex(ValueError, "group identical"):
                audit.reconcile()

    def test_numeric_pairing_prerequisites_cannot_be_relaxed(self):
        with self.mutate("cl9b_h1_acceptance_contract.json",
                         lambda d: d["numeric_gate"].update(same_initial_weights_required=False)):
            with self.assertRaisesRegex(ValueError, "numeric pairing"):
                audit.reconcile()
        with self.mutate("cl6a_readiness_protocol.json",
                         lambda d: d["future_design"].update(same_data_order_per_seed=False)):
            with self.assertRaisesRegex(ValueError, "paired training"):
                audit.reconcile()


if __name__ == "__main__":
    unittest.main()
