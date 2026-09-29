"""No-compute integration test: registered draft -> compiler -> human review -> approved job.

Uses a temporary working directory. Never invokes the Q.ANT experiment runner.
"""
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from scripts.link_research_proposal_outcomes import link

ROOT = Path(__file__).resolve().parents[1]
COMPILER = [sys.executable, "-m", "scripts.compile_ai_researcher_proposal"]
BRIDGE = [sys.executable, "-m", "scripts.materialize_ai_approved_job"]


class Dev3IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.cwd = Path(self.temp.name)
        self.pid = "research-20260929T123456000000Z-" + "a" * 32
        self.draft = self.cwd / "research_queue/ai_researcher/latest.json"
        self.archive = self.cwd / "research_queue/ai_researcher/proposals" / (self.pid + ".json")
        self.compiled = self.cwd / "research_queue/compiled/latest.json"
        self.review = self.cwd / "research_queue/human_review/latest.json"
        self.job_id = "ai-local-width-w9-100epoch-201-202-203-204-205"
        self.job = self.cwd / "research_queue/jobs" / (self.job_id + ".json")
        proposal = {
            "proposal_id": self.pid, "status": "draft_requires_human_review",
            "proposal": {
                "title": "Synthetic provenance integration test",
                "requires_human_approval": True,
                "requested_budget": {
                    "max_parameters": 9000, "max_candidates": 2, "max_epochs": 100,
                    "max_train_samples": 100, "max_test_samples": 100
                },
                "experiment_design": {
                    "dataset": {"name": "ECG200", "train_shape": [100, 96], "test_shape": [100, 96]},
                    "candidates": [{"local_width": 9}],
                    "seeds": [201, 202, 203, 204, 205], "epochs": 100,
                    "procedure": {"same_preregistered_seeds": True, "train_samples": 100,
                                  "test_samples": 100, "aggregate_predictions_per_model": 500}
                },
                "success_criteria": {
                    "gate": "alpha5_w9.aggregate_qant_correct >= alpha5_w16_control.aggregate_qant_correct - 10"
                }
            }
        }
        self.raw = (json.dumps(proposal, indent=2) + "\n").encode()
        self.digest = hashlib.sha256(self.raw).hexdigest()
        for path in (self.draft, self.archive, self.compiled, self.review, self.job):
            path.parent.mkdir(parents=True, exist_ok=True)
        self.draft.write_bytes(self.raw)
        self.archive.write_bytes(self.raw)

    def run_module(self, command, success=True):
        result = subprocess.run(command, cwd=self.cwd, capture_output=True, text=True,
                                env={"PYTHONPATH": str(ROOT)})
        if success:
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        else:
            self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def compile(self):
        self.run_module(COMPILER)
        result = json.loads(self.compiled.read_text())
        self.assertEqual(result["status"], "validated_executable_template")
        self.assertEqual(result["proposal_id"], self.pid)
        self.assertEqual(result["proposal_sha256"], self.digest)

    def approve(self, **changes):
        review = {"decision": "approve", "proposal_id": self.pid, "proposal_sha256": self.digest}
        review.update(changes)
        self.review.write_text(json.dumps(review))

    def test_end_to_end_approved_job_has_archived_lineage(self):
        self.compile()
        self.run_module(BRIDGE, success=False)  # No human approval yet.
        self.assertFalse(self.job.exists())
        self.approve()
        self.run_module(BRIDGE)
        job = json.loads(self.job.read_text())
        self.assertEqual((job["proposal_id"], job["proposal_sha256"]), (self.pid, self.digest))
        self.assertEqual(job["source_compilation"], "research_queue/compiled/latest.json")
        self.assertEqual(job["engine_config"]["candidate_local_width"], 9)

    def synthetic_result(self):
        """A labelled fixture, NOT an observed Q.ANT run or benchmark evidence."""
        return {
            "proposal_id": self.pid, "proposal_sha256": self.digest,
            "experiment_id": "synthetic-dev3-linkage-fixture",
            "dataset": {"name": "ECG200", "train_shape": [100, 96], "test_shape": [100, 96]},
            "configuration": {
                "seeds": [201, 202, 203, 204, 205], "epochs": 100,
                "learning_rate": 0.001, "batch_size": 32, "frequencies": [1, 2],
                "candidates": {"alpha5_w9": [9, 3], "alpha5_w16_control": [16, 3]}
            },
            "summary": {
                "alpha5_w9": {"aggregate_qant_correct": 440, "parameter_count": 5000},
                "alpha5_w16_control": {"aggregate_qant_correct": 445, "parameter_count": 8550}
            },
            "rows": [{"candidate": name, "seed": seed}
                     for name in ("alpha5_w9", "alpha5_w16_control")
                     for seed in (201, 202, 203, 204, 205)]
        }

    def test_dry_run_through_explicit_result_linker(self):
        self.compile()
        self.approve()
        self.run_module(BRIDGE)
        job = json.loads(self.job.read_text())
        result_path = self.cwd / "synthetic_result.json"
        result = self.synthetic_result()
        # Synthetic data exercises provenance and schema only; no training or
        # scientific performance claim is made from these invented counts.
        result["proposal_id"] = job["proposal_id"]
        result["proposal_sha256"] = job["proposal_sha256"]
        result_path.write_text(json.dumps(result))
        manifest = [{"proposal_id": job["proposal_id"],
                     "proposal_file": str(self.archive),
                     "proposal_sha256": job["proposal_sha256"],
                     "strategy": "synthetic-integration-fixture"}]
        report = link(manifest, [result_path])
        self.assertEqual(report["excluded_results"], [])
        linked = report["linked_proposals"][0]
        self.assertEqual(linked["status"], "linked")
        self.assertEqual(linked["proposal_sha256"], self.digest)
        self.assertEqual(linked["results"][0]["experiment_id"],
                         "synthetic-dev3-linkage-fixture")
        self.assertEqual(linked["results"][0]["observations"][0]["correct_margin"], -5)

        # A plausible-looking result must never be joined using ID alone.
        result["proposal_sha256"] = "0" * 64
        result_path.write_text(json.dumps(result))
        rejected = link(manifest, [result_path])
        self.assertEqual(rejected["linked_proposals"][0]["status"], "no_verified_result")
        self.assertEqual(rejected["excluded_results"][0]["reason"],
                         "missing_or_mismatched_result_proposal_digest")

        # Even with matching ID/digest, a different protocol is excluded.
        result["proposal_sha256"] = self.digest
        result["configuration"]["epochs"] = 50
        result_path.write_text(json.dumps(result))
        rejected = link(manifest, [result_path])
        self.assertEqual(rejected["linked_proposals"][0]["status"], "no_verified_result")
        self.assertEqual(rejected["excluded_results"][0]["reason"],
                         "proposal_result_protocol_mismatch")

    def test_modified_archive_rejected_by_compiler(self):
        self.archive.write_bytes(self.raw + b" ")
        self.run_module(COMPILER, success=False)
        self.assertEqual(json.loads(self.compiled.read_text())["status"], "rejected_before_compute")

    def test_changed_compiled_binding_rejected_by_bridge(self):
        self.compile()
        self.approve()
        data = json.loads(self.compiled.read_text())
        data["proposal_sha256"] = "0" * 64
        self.compiled.write_text(json.dumps(data))
        self.run_module(BRIDGE, success=False)
        self.assertFalse(self.job.exists())

    def test_changed_human_approval_rejected_by_bridge(self):
        self.compile()
        self.approve(proposal_id="another-proposal")
        self.run_module(BRIDGE, success=False)
        self.assertFalse(self.job.exists())


if __name__ == "__main__":
    unittest.main()
