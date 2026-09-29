"""Offline v3 protocol and untrusted-result shape tests. All scores are SYNTHETIC."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.research_memory_v3 import search_experiments
from scripts.dev4_round2_isolated_feedback import git_blob_sha
from scripts.scientific_reasoning_v3 import _encoded
from scripts.v3_experiment_protocol import (
    NEW_SEEDS, PROTOCOL_PATH, PRIOR_WIDTH10_REL_PATH, PRIOR_WIDTH10_BLOB_SHA1,
    WIDTHS, inspect_future_result_structure, load_protocol, protocol_sha256,
    protocol_template, prior_width10_reference,
)
from scripts import v3_experiment_protocol as protocol_module


def synthetic_submission():
    """Not a real training output: fake rows only, with invented source hashes."""
    rows = []
    for seed in NEW_SEEDS:
        control = 80 + (seed % 3)
        for width, correct, parameters in (
                (10, control - 1, 5200),
                (11, control + 1, 5800),
                (16, control, 8550)):
            rows.append({
                "seed": seed, "local_width": width,
                "correct_of_100": correct, "parameter_count": parameters,
            })
    return {
        "schema_version": 1,
        "record_kind": "v3_unverified_external_seed_results",
        "protocol_id": load_protocol()["protocol_id"],
        "protocol_sha256": protocol_sha256(),
        "source": {
            "backend": "qant-cpu/software-simulation",
            "workflow_run_id": 99999999999,  # invented for this synthetic test only
            "approval_record_sha256": "a" * 64,
            "raw_artifact_sha256": "b" * 64,
        },
        "rows": rows,
    }


class UnapprovedV3ProtocolContract(unittest.TestCase):
    def test_protocol_reproducibly_matches_pinned_hypothesis_and_old_reference(self):
        before = search_experiments(strategy="gpt-oss-120b")
        with patch("urllib.request.urlopen",
                   side_effect=AssertionError("network forbidden")):
            p = load_protocol()
            self.assertEqual(_encoded(p), _encoded(protocol_template()))
            self.assertEqual(p, load_protocol())
            self.assertEqual(len(protocol_sha256()), 64)
        self.assertEqual(before, search_experiments(strategy="gpt-oss-120b"))
        self.assertEqual(p["status"], "draft_unapproved_unexecuted")
        self.assertEqual(p["record_kind"], "standalone_v3_protocol_draft")
        self.assertEqual([h["local_width"] for h in p["hypotheses"]], [10, 11])
        self.assertEqual(p["method"]["control_width"], 16)
        self.assertEqual(p["method"]["candidate_roles"]["10"],
                         "replication_of_prior_repository_width10_configuration_new_training_seeds")
        self.assertEqual(p["method"]["candidate_roles"]["11"],
                         "new_v3_hypothesis_not_completed_in_isolated_own_strategy_memory")
        self.assertEqual(p["method"]["exploratory_training_seeds"], list(NEW_SEEDS))
        self.assertEqual(p["method"]["expected_model_fits_if_separately_approved"], 15)
        self.assertEqual(p["method"]["backend"], "not_selected")
        self.assertIsNone(p["execution"]["selected_backend"])
        self.assertIsNone(p["execution"]["approval_record"])
        self.assertIsNone(p["execution"]["runnable_workflow"])
        self.assertEqual(p["execution"]["training_runs"], 0)
        self.assertEqual(p["execution"]["groq_calls"], 0)
        self.assertFalse(p["execution"]["modifies_dev4"])
        self.assertTrue(p["execution"]["requires_separate_human_compute_approval"])
        self.assertFalse(p["evaluation_data"]["new_independent_holdout"])
        self.assertEqual(p["evaluation_data"]["split"], "existing_ECG200_test_set")
        self.assertEqual(set(WIDTHS), {10, 11, 16})
        self.assertTrue(set(NEW_SEEDS).isdisjoint({301, 302, 303, 304, 305}))
        self.assertTrue(set(NEW_SEEDS).isdisjoint({201, 202, 203, 204, 205}))

    def test_preexisting_width10_result_is_pinned_and_not_promoted_into_dev4_memory(self):
        before = search_experiments(strategy="gpt-oss-120b")
        prior = prior_width10_reference()
        self.assertEqual(prior, load_protocol()["prior_width10_repository_result"])
        self.assertEqual(prior["source_git_blob_sha1"], PRIOR_WIDTH10_BLOB_SHA1)
        self.assertEqual(prior["backend"], "qant-cpu/software-simulation")
        self.assertEqual(prior["seeds"], [201, 202, 203, 204, 205])
        self.assertEqual(prior["candidate_width"], 10)
        self.assertEqual(prior["candidate_correct_out_of_500"], 444)
        self.assertEqual(prior["paired_control_correct_out_of_500"], 453)
        self.assertEqual(prior["paired_correct_difference"], -9)
        self.assertEqual(prior["candidate_parameter_count"], 5346)
        self.assertEqual(prior["paired_control_parameter_count"], 8550)
        self.assertEqual(search_experiments(strategy="gpt-oss-120b"), before)
        self.assertTrue(all(x["local_width"] != 10 for x in before["results"]))
        self.assertIn("new training seeds", load_protocol()["method"]["candidate_roles"]["10"])

    def test_prior_width10_source_must_match_pin_even_if_summary_claim_is_modified(self):
        path = protocol_module.ROOT / PRIOR_WIDTH10_REL_PATH
        original = path.read_bytes()
        self.assertEqual(git_blob_sha(original), PRIOR_WIDTH10_BLOB_SHA1)
        with tempfile.TemporaryDirectory() as tmp:
            new_path = Path(tmp) / PRIOR_WIDTH10_REL_PATH
            new_path.parent.mkdir(parents=True)
            doc = json.loads(original)
            doc["summary"]["alpha5_w10"]["aggregate_qant_correct"] = 499
            changed = (json.dumps(doc, indent=2) + "\\n").encode()
            new_path.write_bytes(changed)
            with patch.object(protocol_module, "ROOT", Path(tmp)):
                with self.assertRaisesRegex(ValueError, "source changed"):
                    prior_width10_reference()
                # Even repinning modified bytes cannot bypass the row-summary check.
                with patch.object(protocol_module, "PRIOR_WIDTH10_BLOB_SHA1",
                                  git_blob_sha(changed)):
                    with self.assertRaisesRegex(ValueError, "row-summary mismatch"):
                        prior_width10_reference()

    def test_fail_closed_if_any_protocol_field_changes(self):
        canonical = load_protocol()
        variants = {
            "backend_selected_without_approval": lambda p: p["method"].update(
                backend="qant-cpu/software-simulation"),
            "runnable_workflow_added": lambda p: p["execution"].update(
                runnable_workflow="train.yml"),
            "approval_claim_added": lambda p: p["execution"].update(
                approval_record="I_APPROVE"),
            "training_runs_claimed": lambda p: p["execution"].update(training_runs=1),
            "different_candidate": lambda p: p["method"].update(candidate_widths=[10, 12]),
            "historical_seeds": lambda p: p["method"].update(
                exploratory_training_seeds=[301, 302, 303, 304, 305]),
            "fabricated_holdout": lambda p: p["evaluation_data"].update(
                new_independent_holdout=True),
            "new_hypothesis_hash": lambda p: p["hypotheses"][0].update(
                proposal_sha256="f" * 64),
            "fake_prior_width10_score": lambda p: p["prior_width10_repository_result"].update(
                candidate_correct_out_of_500=490),
            "promote_prior_result_to_memory": lambda p: p["prior_width10_repository_result"].update(
                classification="completed_v3_hypothesis_evidence"),
            "unexpected_extra_field": lambda p: p.update(execute=True),
            "false_instead_of_zero": lambda p: p["execution"].update(training_runs=False),
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "altered.json"
            for name, mutate in variants.items():
                with self.subTest(name=name):
                    trial = copy.deepcopy(canonical)
                    mutate(trial)
                    path.write_text(json.dumps(trial))
                    with self.assertRaisesRegex(ValueError, "differs"):
                        load_protocol(path)

    def test_synthetic_future_rows_only_get_structural_unverified_summary(self):
        with patch("urllib.request.urlopen",
                   side_effect=AssertionError("real network forbidden")):
            summary = inspect_future_result_structure(synthetic_submission())
        self.assertEqual(summary["status"], "structural_checks_only_unverified")
        self.assertEqual(summary["protocol_sha256"], protocol_sha256())
        self.assertEqual(summary["training_runs_started_by_validator"], 0)
        self.assertEqual(summary["repository_writes"], 0)
        self.assertFalse(summary["authenticity_verified"])
        self.assertFalse(summary["compute_approval_verified"])
        self.assertFalse(summary["eligible_for_research_memory"])
        self.assertFalse(summary["independent_holdout_data"])
        self.assertTrue(summary["requires_separate_human_provenance_review"])
        by_width = {r["candidate_width"]: r for r in summary["comparisons"]}
        self.assertEqual(set(by_width), {10, 11})
        self.assertEqual(by_width[10]["total_paired_correct_difference_unverified"], -5)
        self.assertEqual(by_width[11]["total_paired_correct_difference_unverified"], 5)
        self.assertEqual(by_width[10]["descriptive_accuracy_relation_unverified"],
                         "below_fresh_control")
        self.assertEqual(by_width[11]["descriptive_accuracy_relation_unverified"],
                         "above_fresh_control")
        for comparison in summary["comparisons"]:
            self.assertEqual(len(comparison["paired_correct_differences_by_seed_unverified"]), 5)
            self.assertEqual(comparison["paired_difference_range_unverified"][0],
                             comparison["paired_difference_range_unverified"][1])
            self.assertEqual(comparison["sample_sd_paired_differences_unverified"], 0)

    def test_reject_incomplete_duplicate_or_forged_external_results(self):
        baseline = synthetic_submission()

        def edit_row(doc, field, value, index=0):
            doc["rows"][index][field] = value

        cases = {
            "missing_seed_width_row": lambda d: d["rows"].pop(),
            "duplicate_seed_width": lambda d: d["rows"].__setitem__(
                1, copy.deepcopy(d["rows"][0])),
            "unexpected_seed_301": lambda d: edit_row(d, "seed", 301),
            "unexpected_width_12": lambda d: edit_row(d, "local_width", 12),
            "false_seed": lambda d: edit_row(d, "seed", True),
            "bool_correct": lambda d: edit_row(d, "correct_of_100", False),
            "negative_correct": lambda d: edit_row(d, "correct_of_100", -1),
            "more_than_100_correct": lambda d: edit_row(d, "correct_of_100", 101),
            "bad_parameter_count": lambda d: edit_row(d, "parameter_count", 0),
            "inconsistent_parameters_same_width": lambda d: edit_row(
                d, "parameter_count", 8888, index=3),
            "extra_pseudo_accuracy": lambda d: d["rows"][0].update(accuracy=0.99),
            "claim_verified": lambda d: d.update(status="verified"),
            "wrong_protocol_sha": lambda d: d.update(protocol_sha256="0" * 64),
            "wrong_protocol_id": lambda d: d.update(protocol_id="dev4-round3"),
            "unverified_optical_backend": lambda d: d["source"].update(
                backend="qant-photonic"),
            "invalid_workflow_run_id": lambda d: d["source"].update(workflow_run_id=True),
            "invalid_approval_hash": lambda d: d["source"].update(
                approval_record_sha256="yes"),
            "additional_source_claim": lambda d: d["source"].update(verified=True),
        }
        for label, change in cases.items():
            with self.subTest(label=label):
                trial = copy.deepcopy(baseline)
                change(trial)
                with self.assertRaises((ValueError, TypeError, KeyError)):
                    inspect_future_result_structure(trial)

    def test_inspection_does_not_write_back_to_frozen_memory_or_draft(self):
        before_evidence = search_experiments(strategy="gpt-oss-120b")
        before_bytes = PROTOCOL_PATH.read_bytes()
        inspect_future_result_structure(synthetic_submission())
        self.assertEqual(PROTOCOL_PATH.read_bytes(), before_bytes)
        self.assertEqual(
            search_experiments(strategy="gpt-oss-120b"), before_evidence)


if __name__ == "__main__":
    unittest.main()
