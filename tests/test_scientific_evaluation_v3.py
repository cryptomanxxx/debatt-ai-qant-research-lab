"""No model/API/compute: scientific evaluation is a pinned-reference plan only."""
import copy
import hashlib
import unittest

from scripts import research_memory_v3 as memory
from scripts.scientific_evaluation_v3 import (
    CORRECT_OUT_OF, build_evaluation_plan,
)
from scripts.scientific_reasoning_v3 import _encoded

PROPOSAL = {
    "hypothesis": "Width 10 could improve ECG200 accuracy in a future controlled experiment.",
    "rationale": "The two verified widths invite a test, not a prediction or proven trend.",
    "local_width": 10,
    "dataset": "ECG200",
    "epochs": 100,
    "evidence_ids": ["dev4-r1-gpt-oss-120b", "dev4-r2-gpt-oss-120b"],
    "expected_measurements": [
        "candidate_correct", "candidate_parameters", "gate_pass",
    ],
}


class EvaluationPlanContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.observed = memory.search_experiments(strategy="gpt-oss-120b")["results"]

    def plan(self, proposal=None, observed=None):
        return build_evaluation_plan(
            proposal=copy.deepcopy(PROPOSAL if proposal is None else proposal),
            observed=copy.deepcopy(self.observed if observed is None else observed),
            strategy="gpt-oss-120b",
        )

    def test_control_thresholds_are_distinct_and_reproducibly_pinned(self):
        plan = self.plan()
        thresholds = plan["thresholds"]
        self.assertEqual(plan["record_kind"], "unexecuted_evaluation_plan")
        self.assertEqual(plan["status"], "planning_only")
        self.assertEqual(thresholds["historical_gate_pass_at_least_correct"], 442)
        self.assertEqual(thresholds["ties_control_exact_correct"], 452)
        self.assertEqual(thresholds["strictly_exceeds_control_at_least_correct"], 453)
        self.assertEqual(thresholds["best_observed_own_strategy_correct"], 445)
        self.assertEqual(thresholds["strictly_exceeds_best_observed_at_least_correct"], 446)
        self.assertLess(
            thresholds["historical_gate_pass_at_least_correct"],
            thresholds["ties_control_exact_correct"])
        self.assertEqual(
            thresholds["strictly_exceeds_control_at_least_correct"],
            thresholds["ties_control_exact_correct"] + 1)
        reference = plan["historical_reference_only"]
        self.assertEqual(reference["control_local_width"], 16)
        self.assertEqual(reference["control_correct"], 452)
        self.assertEqual(reference["control_parameters"], 8550)
        self.assertEqual(reference["correct_out_of"], CORRECT_OUT_OF)
        self.assertEqual(reference["gate_margin_correct"], 10)
        self.assertEqual(len(reference["reference_rounds"]), 2)
        for item in reference["reference_rounds"]:
            self.assertEqual(len(item["git_blob_sha1"]), 40)
            self.assertEqual(len(item["original_artifact_sha256"]), 64)
        self.assertEqual(plan["proposal_sha256"], hashlib.sha256(_encoded(PROPOSAL)).hexdigest())

    def test_candidate_is_unmeasured_and_human_authorization_required(self):
        plan = self.plan()
        candidate = plan["candidate"]
        self.assertEqual(candidate["dataset"], "ECG200")
        self.assertEqual(candidate["local_width"], 10)
        self.assertEqual(candidate["epochs"], 100)
        self.assertEqual(candidate["comparison_seeds"], [301, 302, 303, 304, 305])
        self.assertFalse(candidate["has_completed_own_strategy_observation"])
        self.assertIsNone(candidate["measured_candidate_correct"])
        self.assertIsNone(candidate["measured_candidate_parameters"])
        self.assertTrue(plan["requires_separate_human_compute_approval"])
        self.assertEqual(plan["training_runs"], 0)
        self.assertEqual(plan["repository_mutations"], 0)
        text = " ".join(plan["interpretation_rules"])
        self.assertIn("not a v3 success criterion", text)
        self.assertIn("undefined", text)
        self.assertIn("cannot be inferred", text)
        self.assertIn("seed-level results", text)

    def test_observed_history_is_only_completed_own_strategy_not_hypothesis(self):
        plan = self.plan()
        self.assertEqual(plan["observed_verified_ids"], [
            "dev4-r1-gpt-oss-120b", "dev4-r2-gpt-oss-120b"])
        self.assertNotIn("dev4-r3-gpt-oss-120b", plan["observed_verified_ids"])
        proposal = copy.deepcopy(PROPOSAL)
        proposal["local_width"] = 9
        self.assertTrue(self.plan(proposal)["candidate"]["has_completed_own_strategy_observation"])
        self.assertIsNone(self.plan(proposal)["candidate"]["measured_candidate_correct"])

    def test_reject_tampered_pinned_records_and_cross_strategy(self):
        for change in ("candidate_correct", "control_correct", "source", "strategy",
                       "duplicate", "missing", "unknown", "wrong_type"):
            with self.subTest(change=change):
                rows = copy.deepcopy(self.observed)
                if change == "candidate_correct":
                    rows[0]["candidate_correct"] = 500
                elif change == "control_correct":
                    rows[0]["control_correct"] = 500
                elif change == "source":
                    rows[0]["source"]["git_blob_sha1"] = "0" * 40
                elif change == "strategy":
                    rows[0]["strategy"] = "random-search"
                elif change == "duplicate":
                    rows.append(copy.deepcopy(rows[0]))
                elif change == "missing":
                    rows = []
                elif change == "unknown":
                    rows[0]["experiment_id"] = "dev4-r3-gpt-oss-120b"
                else:
                    rows[0] = "not a record"
                with self.assertRaises(ValueError):
                    self.plan(observed=rows)

    def test_reject_unobserved_citations_and_invalid_experiment_configuration(self):
        cases = []
        proposal = copy.deepcopy(PROPOSAL)
        proposal["evidence_ids"] = ["dev4-r1-random-search"]
        cases.append(proposal)
        proposal = copy.deepcopy(PROPOSAL)
        proposal["local_width"] = 16
        cases.append(proposal)
        proposal = copy.deepcopy(PROPOSAL)
        proposal["epochs"] = 50
        cases.append(proposal)
        proposal = copy.deepcopy(PROPOSAL)
        proposal["dataset"] = "other"
        cases.append(proposal)
        for proposal in cases:
            with self.subTest(proposal=proposal), self.assertRaises(ValueError):
                self.plan(proposal)
        with self.assertRaisesRegex(ValueError, "unobserved"):
            self.plan(observed=self.observed[:1])

    def test_plan_does_not_modify_frozen_sources_or_input(self):
        originals = {
            filename: (memory.BENCH / filename).read_bytes()
            for _, filename in memory.ROUNDS
        }
        observed_before = _encoded(self.observed)
        proposal_before = _encoded(PROPOSAL)
        self.plan()
        self.assertEqual(_encoded(self.observed), observed_before)
        self.assertEqual(_encoded(PROPOSAL), proposal_before)
        for filename, original in originals.items():
            self.assertEqual((memory.BENCH / filename).read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
