"""CL-10C artifact-format tests using hand-built schema fixtures, not model runs."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from cl10c_artifact_writer import build_payloads, write_artifact_bundle

H = "a" * 64
COMMIT = "b" * 40


def manifest():
    return {
        "artifact_schema": "cl10c-v1",
        "immutable_source_commit_sha": COMMIT,
        "protocol_sha256": H,
        "source_tree_sha256": H,
        "environment_manifest_sha256": H,
        "dataset_manifest_sha256": H,
        "units_and_metric_definitions_sha256": H,
        "instrumentation_version_sha256": H,
        "per_seed_dataset_sha256": {str(i): H for i in range(101, 106)},
        "ordered_seed_list": list(range(101, 106)),
        "ordered_arm_list": [
            "dense_sgd", "local_hybrid",
            "sparse_local_hybrid", "sparse_shared_index",
        ],
        "task_splits_and_digest_list": [{"task": "T1", "digest": H}],
        "review_id": "fixture-review-not-real",
        "explicit_user_authorization_reference": "fixture-not-real-approval",
        "physical_energy": "NOT_MEASURED",
    }


def attempt(status="COMPLETE"):
    return {
        "run_id": "fixture-run", "arm": "dense_sgd", "seed": 101,
        "task": "T1", "checkpoint": "post_T1", "phase": "train",
        "attempt_number": 1, "run_status": status,
        "failure_reason": None if status == "COMPLETE" else "fixture failure",
        "source_commit_sha": COMMIT, "observed_record_digest": None,
        "artifact_digest_or_null": None,
    }


def bundle_args():
    return {
        "manifest": manifest(),
        "attempts": [attempt()],
        "accuracy": [{
            "arm": "dense_sgd", "seed": 101, "task": "T1",
            "checkpoint": "post_T1", "phase": "heldout_inference",
            "correct": 33, "eligible": 64,
        }],
        "events": [],
        "failures": [],
    }


class TestCL10C(unittest.TestCase):
    def test_bundle_checksums_and_no_overwrite(self):
        args = bundle_args()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundle = write_artifact_bundle(root, "fixture-001", **args)
            expected = {
                "manifest.json", "attempts.jsonl", "heldout_accuracy.jsonl",
                "source_events.jsonl", "failure_ledger.jsonl",
                "checksums.sha256",
            }
            self.assertEqual({x.name for x in bundle.iterdir()}, expected)
            for line in (bundle / "checksums.sha256").read_text().splitlines():
                digest, filename = line.split("  ")
                self.assertEqual(
                    hashlib.sha256((bundle / filename).read_bytes()).hexdigest(),
                    digest,
                )
            with self.assertRaises(ValueError):
                write_artifact_bundle(root, "fixture-001", **args)
            self.assertEqual(len(list(root.iterdir())), 1)

    def test_failures_must_be_in_both_ledgers(self):
        args = bundle_args()
        args["attempts"] = [attempt("FAILED")]
        with self.assertRaises(ValueError):
            build_payloads(**args)
        args["failures"] = [attempt("FAILED")]
        build_payloads(**args)

    def test_invalid_values_fail_closed(self):
        changes = (
            ("manifest", "physical_energy", 3.7),
            ("manifest", "immutable_source_commit_sha", H),
            ("manifest", "review_id", None),
        )
        for group, field, value in changes:
            args = bundle_args()
            args[group][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                build_payloads(**args)
        args = bundle_args()
        args["accuracy"][0]["correct"] = 65
        with self.assertRaises(ValueError):
            build_payloads(**args)
        args = bundle_args()
        args["attempts"].append(dict(args["attempts"][0]))
        with self.assertRaises(ValueError):
            build_payloads(**args)
        args = bundle_args()
        args["events"] = [{
            "arm": "dense_sgd", "seed": 101, "task": "T1",
            "checkpoint": "post_T1", "phase": "train",
            "denominator": 128, "provenance": "STATIC_ESTIMATE",
            "logical_parameter_reads": 0, "logical_parameter_writes": 0,
            "input_visits": 0, "active_index_visits": 0,
            "control_events": 0, "nonlinear_events": 0,
        }]
        with self.assertRaises(ValueError):
            build_payloads(**args)

    def test_unsafe_run_id_rejected_before_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            for run_id in ("../escape", "/absolute", "fixture/child", ""):
                with self.subTest(run_id=run_id), self.assertRaises(ValueError):
                    write_artifact_bundle(Path(tmp), run_id, **bundle_args())
            self.assertEqual(list(Path(tmp).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
