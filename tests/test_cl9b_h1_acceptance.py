"""CL-9B H1 acceptance contract: static review only."""
import json
from pathlib import Path
import unittest

BASE = Path(__file__).resolve().parents[1] / "integrations/continual_learning"


class CL9BAcceptanceTests(unittest.TestCase):
    def test_matches_preexisting_h1_contrasts_and_quality_guards(self):
        record = json.loads((BASE / "cl9b_h1_acceptance_contract.json").read_text())
        protocol = json.loads((BASE / record["source_protocol"]).read_text())
        self.assertEqual(record["decision"], "NO_GO")
        self.assertTrue(record["review_only"])
        self.assertEqual(record["status"], "PROPOSED_UNREVIEWED_THRESHOLDS")
        self.assertEqual([c["id"] for c in record["contrasts"]],
                         ["primary_sparse", "secondary_index_reuse"])
        for contrast in record["contrasts"]:
            expected = next(c for c in protocol["comparisons"] if c["id"] == contrast["id"])
            self.assertEqual(contrast["reference"], expected["left"])
            self.assertEqual(contrast["candidate"], expected["right"])
        self.assertEqual(record["descriptive_seeds"], protocol["future_design"]["seeds"])
        self.assertEqual(record["quality_gate"]["final_mean_accuracy_floor_per_seed_and_arm"],
                         protocol["preregistered_quality"]["final_mean_accuracy_floor"])
        self.assertEqual(record["quality_gate"]["maximum_absolute_final_mean_paired_accuracy_gap_per_seed"],
                         protocol["preregistered_quality"]["paired_accuracy_difference_max_abs"])
        self.assertEqual(record["work_metrics"]["physical_energy"], "NOT_MEASURED")

    def test_no_go_and_three_exclusive_descriptive_outcomes(self):
        record = json.loads((BASE / "cl9b_h1_acceptance_contract.json").read_text())
        self.assertEqual({x["outcome"] for x in record["pilot_decision_rules"]},
                         {"DESCRIPTIVE_H1_SUPPORT", "NO_OBSERVED_H1_ADVANTAGE", "INCONCLUSIVE"})
        for flag in ("model_training_authorized", "model_inference_authorized",
                     "dataset_generation_authorized", "energy_measured"):
            self.assertIs(record[flag], False)
        readiness = json.loads((BASE / record["source_readiness"]).read_text())
        self.assertEqual(record["gates_still_open"],
                         [g["id"] for g in readiness["minimum_gates"]])
        self.assertTrue(all(g["status"] == "OPEN" for g in readiness["minimum_gates"]))
        self.assertEqual(record["numeric_gate"]["status"], "OPEN")
        self.assertEqual(record["work_metrics"]["unit_cost_weights"],
                         "UNRESOLVED_INDEPENDENT_REVIEW")


if __name__ == "__main__":
    unittest.main()
