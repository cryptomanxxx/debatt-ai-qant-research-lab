"""v3 hypothesis registry/protocol replay: completely offline, no training."""
import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import research_memory_v3 as memory
from scripts.dev4_round2_isolated_feedback import git_blob_sha
from scripts.scientific_reasoning_v3 import _encoded
from scripts.v3_hypothesis_registry import (
    OBSERVED_IDS, REGISTRY, SOURCE_PINS, _pinned_protocol,
    build_offline_study_draft, load_hypotheses, verify_original_report_bytes,
)


class ArchivedHypothesisContract(unittest.TestCase):
    def test_both_original_proposals_replay_to_unexecuted_plans(self):
        before = memory.search_experiments(strategy="gpt-oss-120b")
        with patch("urllib.request.urlopen",
                   side_effect=AssertionError("network must stay disabled")):
            archive = load_hypotheses()
            first = build_offline_study_draft()
            second = build_offline_study_draft()
        self.assertEqual(_encoded(first), _encoded(second))
        self.assertEqual([x["proposal"]["local_width"] for x in archive["entries"]],
                         [11, 10])
        self.assertEqual({x["hypothesis_id"] for x in archive["entries"]},
                         set(SOURCE_PINS))
        self.assertEqual(first["record_kind"], "offline_study_draft")
        self.assertEqual(first["status"], "unapproved_unexecuted")
        self.assertTrue(first["requires_separate_human_compute_approval"])
        self.assertFalse(first["modifies_frozen_dev4"])
        self.assertEqual(first["groq_calls"], 0)
        self.assertEqual(first["training_runs"], 0)
        self.assertEqual(first["proposed_future_design_not_authorized"]["candidate_widths"],
                         [10, 11])
        self.assertEqual(first["proposed_future_design_not_authorized"]["backend_status"],
                         "not_selected")
        self.assertEqual(len(first["proposals"]), 2)
        for entry in first["proposals"]:
            plan = entry["evaluation_plan"]
            self.assertEqual(entry["state"], "proposed_untested")
            self.assertEqual(plan["record_kind"], "unexecuted_evaluation_plan")
            self.assertEqual(plan["status"], "planning_only")
            self.assertEqual(plan["training_runs"], 0)
            self.assertEqual(plan["repository_mutations"], 0)
            self.assertTrue(plan["requires_separate_human_compute_approval"])
            self.assertIsNone(plan["candidate"]["measured_candidate_correct"])
            self.assertIsNone(plan["candidate"]["measured_candidate_parameters"])
            self.assertEqual(plan["thresholds"][
                "historical_gate_pass_at_least_correct"], 442)
            self.assertEqual(plan["thresholds"]["ties_control_exact_correct"], 452)
            self.assertEqual(plan["thresholds"][
                "strictly_exceeds_control_at_least_correct"], 453)
            self.assertEqual(plan["thresholds"][
                "best_observed_own_strategy_correct"], 445)
            self.assertEqual(plan["observed_verified_ids"], sorted(OBSERVED_IDS))
        self.assertEqual(memory.search_experiments(
            strategy="gpt-oss-120b"), before)

    def test_pinned_historical_settings_and_proposed_paired_seeds(self):
        draft = build_offline_study_draft()
        history = draft["historical_protocol_reference"]
        self.assertEqual(history["epochs"], 100)
        self.assertEqual(history["seeds"], [301, 302, 303, 304, 305])
        self.assertEqual(history["learning_rate"], 0.001)
        self.assertEqual(history["batch_size"], 32)
        self.assertEqual(history["frequencies"], [1, 2])
        self.assertEqual(history["control_local_width"], 16)
        self.assertEqual(history["source_backend_only"], "qant-cpu/software-simulation")
        self.assertEqual(history["control_comparison_correct"], 452)
        self.assertEqual(history["control_comparison_parameters"], 8550)
        design = draft["proposed_future_design_not_authorized"]
        self.assertEqual(design["paired_control_width"], 16)
        self.assertTrue(design["fresh_paired_control_proposed"])
        self.assertFalse(design["automatic_compute_or_dispatch"])
        self.assertEqual(design["primary_endpoint_status"], "requires_explicit_preregistration")
        self.assertIn("paired_correct_difference", design["seed_level_fields"])
        self.assertEqual(set(_pinned_protocol()[1]), set(history["seeds"]))

    def test_registry_edits_cannot_promote_hypotheses_to_measurements(self):
        original = load_hypotheses()
        scenarios = {
            "promoted_status": lambda d: d["entries"][0].update(state="completed"),
            "invented_score": lambda d: d["entries"][1].update(
                measured_candidate_correct=490),
            "fabricated_result_id": lambda d: d["entries"][0].update(
                completed_experiment_id="v3-exp-width11"),
            "proposal_changed": lambda d: d["entries"][0]["proposal"].update(
                local_width=12),
            "source_hash_changed": lambda d: d["entries"][0]["source"].update(
                original_report_sha256="0" * 64),
            "invented_citation": lambda d: d["entries"][1]["proposal"].update(
                evidence_ids=["dev4-r1-random-search"]),
            "missing_original": lambda d: d["entries"].pop(),
            "duplicate_original": lambda d: d["entries"].append(copy.deepcopy(d["entries"][0])),
            "altered_review": lambda d: d["entries"][0]["review"].update(
                assessment="testable_with_caveats"),
            "boolean_training": lambda d: d.update(training_runs=False),
        }
        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp) / "registry.json"
            for label, mutate in scenarios.items():
                with self.subTest(scenario=label):
                    doc = copy.deepcopy(original)
                    mutate(doc)
                    file.write_text(json.dumps(doc, ensure_ascii=False))
                    with self.assertRaises((ValueError, TypeError, KeyError)):
                        load_hypotheses(file)
                    with self.assertRaises((ValueError, TypeError, KeyError)):
                        build_offline_study_draft(file)

    def test_report_import_requires_exact_original_hash_and_metadata(self):
        archive = load_hypotheses()
        evidence = memory.search_experiments(strategy="gpt-oss-120b")["results"]
        for item in archive["entries"]:
            key = item["hypothesis_id"]
            pin = SOURCE_PINS[key]
            source = {
                "audit": [
                    {"kind": "proposal",
                     "model_output_sha256": pin["proposal_output_sha256"]},
                ],
                "result": {
                    "status": "proposal_only", "strategy": "gpt-oss-120b",
                    "proposal": item["proposal"], "evidence": evidence,
                    "requires_separate_human_compute_approval": True,
                },
                "training_runs": 0,
            }
            if pin["reviewed"]:
                source["observed_evidence"] = evidence
                source["review_model_calls"] = 1
                source["result"]["critical_review"] = {
                    "assessment": item["review"]["assessment"],
                    "reviewed_proposal_sha256": pin["proposal_sha256"],
                }
            raw = (json.dumps(source, ensure_ascii=False) + "\n").encode()
            # Simulate an independently authenticated source file with a
            # different byte sequence; no real original is bundled or fetched.
            local_pin = dict(pin, report_sha256=hashlib.sha256(raw).hexdigest())
            with self.subTest(hypothesis_id=key), patch.dict(
                    SOURCE_PINS, {key: local_pin}):
                self.assertTrue(verify_original_report_bytes(raw, item))
                with self.assertRaisesRegex(ValueError, "fingerprint"):
                    verify_original_report_bytes(raw + b" ", item)
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                verify_original_report_bytes(raw, item)

    def test_reject_drift_in_historical_protocol(self):
        filename = memory.ROUNDS[0][1]
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            for _, name in memory.ROUNDS:
                (tmpdir / name).write_bytes((memory.BENCH / name).read_bytes())
            changed = json.loads((tmpdir / filename).read_bytes())
            changed["evaluation"]["learning_rate"] = 0.002
            raw = (json.dumps(changed, indent=2) + "\n").encode()
            (tmpdir / filename).write_bytes(raw)
            # Even if the Git blob pin were independently advanced, the
            # declared historical settings must still fail this v3 contract.
            with patch.object(memory, "BENCH", tmpdir), patch.dict(
                    "scripts.v3_hypothesis_registry.PINNED_BLOBS",
                    {filename: git_blob_sha(raw)}):
                with self.assertRaisesRegex(ValueError, "hyperparameters"):
                    _pinned_protocol()

    def test_actual_original_source_fingerprints_are_documented(self):
        doc = load_hypotheses()
        self.assertEqual(
            [x["source"]["workflow_run_id"] for x in doc["entries"]],
            [36597959269, 36601207596])
        self.assertEqual(
            [x["source"]["original_report_sha256"] for x in doc["entries"]],
            [SOURCE_PINS[x["hypothesis_id"]]["report_sha256"]
             for x in doc["entries"]])
        self.assertTrue(all(x["completed_experiment_id"] is None
                            for x in doc["entries"]))


if __name__ == "__main__":
    unittest.main()
