"""Tests for explicit proposal/result provenance; no guessed joins."""
import json
import tempfile
import unittest
from pathlib import Path
from scripts.link_research_proposal_outcomes import link


class LinkageTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.proposal = self.root / "proposal.json"
        self.result = self.root / "result.json"
        self.design = {"dataset": {"name": "ECG200", "train_shape": [100, 96], "test_shape": [100, 96]},
                       "seeds": [1, 2, 3, 4, 5], "epochs": 100, "candidates": [{"local_width": 8}]}
        self.proposal.write_text(json.dumps({"proposal_id": "p1", "proposal": {"experiment_design": self.design}}))
        self.data = {"proposal_id": "p1", "experiment_id": "e1", "dataset": self.design["dataset"],
                     "configuration": {"seeds": self.design["seeds"], "epochs": 100, "learning_rate": .001,
                                       "batch_size": 32, "frequencies": [1, 2],
                                       "candidates": {"alpha5_w8": [8, 3], "alpha5_w16_control": [16, 3]}},
                     "summary": {"alpha5_w8": {"aggregate_qant_correct": 444, "parameter_count": 4278},
                                 "alpha5_w16_control": {"aggregate_qant_correct": 448, "parameter_count": 8550}}}
        self.manifest = [{"proposal_id": "p1", "proposal_file": str(self.proposal), "strategy": "gpt-oss"}]

    def run_link(self):
        self.result.write_text(json.dumps(self.data))
        return link(self.manifest, [self.result])

    def test_valid_explicit_link(self):
        row = self.run_link()["linked_proposals"][0]
        self.assertEqual(row["status"], "linked")
        self.assertEqual(row["results"][0]["observations"][0]["correct_margin"], -4)

    def test_no_guessed_link_from_similar_experiment(self):
        self.data["proposal_id"] = "different"
        report = self.run_link()
        self.assertEqual(report["linked_proposals"][0]["status"], "no_verified_result")
        self.assertEqual(report["excluded_results"][0]["reason"], "unknown_or_missing_proposal_id")

    def test_protocol_mismatch_excluded(self):
        self.data["configuration"]["epochs"] = 50
        self.assertEqual(self.run_link()["excluded_results"][0]["reason"], "proposal_result_protocol_mismatch")

    def test_candidate_mismatch_excluded(self):
        self.data["configuration"]["candidates"]["alpha5_w8"] = [12, 3]
        self.assertEqual(self.run_link()["excluded_results"][0]["reason"], "candidate_width_mismatch")

    def test_incomplete_result_excluded(self):
        del self.data["configuration"]["learning_rate"]
        self.assertEqual(self.run_link()["excluded_results"][0]["reason"], "incomplete_or_invalid_paired_result")

    def test_duplicate_manifest_rejected(self):
        with self.assertRaises(ValueError):
            link(self.manifest * 2, [])

    def test_proposal_id_mismatch_rejected(self):
        self.manifest[0]["proposal_id"] = "p2"
        with self.assertRaises(ValueError):
            link(self.manifest, [])


if __name__ == "__main__":
    unittest.main()
