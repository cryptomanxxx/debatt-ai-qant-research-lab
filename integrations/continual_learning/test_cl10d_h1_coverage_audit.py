"""CL-10D tests: fabricated records only; no model/data generation."""
import tempfile
import unittest
from pathlib import Path

from cl10c_artifact_writer import write_artifact_bundle
from cl10d_h1_coverage_audit import audit_existing_bundle, validate_complete_h1
from test_cl10c_artifact_writer import H, COMMIT, manifest

ARMS = ("dense_sgd", "local_hybrid", "sparse_local_hybrid", "sparse_shared_index")
SEEDS = range(101, 106)
TASKS = ("T1", "T2", "T3", "T4")
CHECKPOINTS = ("pretrain", "post_T1", "post_T2", "post_T3", "post_T4")


def fixtures():
    """Hand-authored schema shapes over a fixed Cartesian grid, not observations."""
    attempts = [
        {
            "run_id": "fixture-coverage", "arm": arm, "seed": seed, "task": task,
            "checkpoint": "post_" + task, "phase": "train",
            "attempt_number": 1, "run_status": "COMPLETE", "failure_reason": None,
            "source_commit_sha": COMMIT, "observed_record_digest": None,
            "artifact_digest_or_null": None,
        }
        for arm in ARMS for seed in SEEDS for task in TASKS
    ]
    accuracy = [
        {
            "arm": arm, "seed": seed, "checkpoint": checkpoint,
            "task": task, "phase": "heldout_inference",
            "correct": 32, "eligible": 64,
        }
        for arm in ARMS for seed in SEEDS
        for checkpoint in CHECKPOINTS for task in TASKS
    ]
    return {
        "manifest": manifest(), "attempts": attempts,
        "accuracy": accuracy, "events": [], "failures": [],
    }


class TestCL10D(unittest.TestCase):
    def test_complete_fixture_and_read_only_checksum_audit(self):
        args = fixtures()
        result = validate_complete_h1(**args)
        self.assertEqual(result["training_attempts"], 80)
        self.assertEqual(result["heldout_records"], 400)
        self.assertFalse(result["authenticity_verified"])
        self.assertFalse(result["source_event_claim_allowed"])
        self.assertEqual(result["decision"], "NO_GO")
        with tempfile.TemporaryDirectory() as tmp:
            root = write_artifact_bundle(Path(tmp), "fixture-complete", **args)
            self.assertEqual(audit_existing_bundle(root), result)
            with (root / "heldout_accuracy.jsonl").open("ab") as stream:
                stream.write(b" ")
            with self.assertRaises(ValueError):
                audit_existing_bundle(root)

    def test_missing_duplicate_and_wrong_size_rejected(self):
        for name in ("attempts", "accuracy"):
            args = fixtures()
            args[name].pop()
            with self.subTest(name=name), self.assertRaises(ValueError):
                validate_complete_h1(**args)
            args = fixtures()
            args[name][-1] = dict(args[name][0])
            with self.subTest(name=name + " duplicate"), self.assertRaises(ValueError):
                validate_complete_h1(**args)
        args = fixtures()
        args["accuracy"][0]["eligible"] = 63
        with self.assertRaises(ValueError):
            validate_complete_h1(**args)

    def test_source_binding_and_retry_rejected(self):
        args = fixtures()
        args["attempts"][0]["source_commit_sha"] = "c" * 40
        with self.assertRaises(ValueError):
            validate_complete_h1(**args)
        args = fixtures()
        args["attempts"][0]["attempt_number"] = 2
        with self.assertRaises(ValueError):
            validate_complete_h1(**args)
        args = fixtures()
        args["attempts"][0]["run_id"] = "another-run"
        with self.assertRaises(ValueError):
            validate_complete_h1(**args)

    def test_incomplete_event_matrix_rejected(self):
        args = fixtures()
        args["events"] = [{
            "arm": "dense_sgd", "seed": 101, "task": "T1",
            "checkpoint": "post_T1", "phase": "train",
            "denominator": 128, "provenance": "OBSERVED_SOURCE_EVENTS",
            "logical_parameter_reads": 0, "logical_parameter_writes": 0,
            "input_visits": 0, "active_index_visits": 0,
            "control_events": 0, "nonlinear_events": 0,
        }]
        with self.assertRaises(ValueError):
            validate_complete_h1(**args)

    def test_failure_cannot_masquerade_as_complete(self):
        args = fixtures()
        args["attempts"][0]["run_status"] = "FAILED"
        args["attempts"][0]["failure_reason"] = "fixture failure"
        args["failures"] = [dict(args["attempts"][0])]
        with self.assertRaises(ValueError):
            validate_complete_h1(**args)


if __name__ == "__main__":
    unittest.main()
