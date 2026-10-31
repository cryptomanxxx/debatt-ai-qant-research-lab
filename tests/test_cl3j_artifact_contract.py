"""Fictional manifest fixtures only, not model runs or research results."""
import copy
import unittest

from integrations.continual_learning.cl3j_artifact_contract import (
    artifact_review_status, validate_future_record,
)
from integrations.continual_learning.cl3e_measurement_ledger import SCOPE_VERSION


def fake_record():
    return {
        "protocol_sha256": "a" * 64,
        "source_commit_sha": "b" * 40,
        "generator_source_sha256": "c" * 64,
        "dataset_content_sha256": "d" * 64,
        "python_version": "3.11.9",
        "model_arm": "dense_sgd",
        "paired_seed": 101,
        "training_update_count": 512,
        "checkpoint_accuracy_matrix": [[0.5] * 4 for _ in range(5)],
        "checkpoint_balanced_accuracy_matrix": [[0.5] * 4 for _ in range(5)],
        "operation_scope_version": SCOPE_VERSION,
        "scientific_review_id": "FAKE-scientific-review",
        "compute_approval_id": "FAKE-compute-approval",
    }


class CL3JArtifactContractTests(unittest.TestCase):
    def test_pure_schema_accepts_complete_fictional_record(self):
        validate_future_record(fake_record())
        summary = artifact_review_status()
        self.assertFalse(summary["training_allowed"])
        self.assertFalse(summary["schema_validation_proves_authenticity"])
        self.assertFalse(summary["physical_energy_inference_allowed"])

    def test_missing_or_mutated_provenance_rejected(self):
        for key, value in (
            ("protocol_sha256", None),
            ("source_commit_sha", "b" * 7),
            ("dataset_content_sha256", "G" * 64),
            ("model_arm", "sparse_local_hybrid"),
            ("paired_seed", True),
            ("training_update_count", 513),
            ("training_update_count", True),
            ("operation_scope_version", "unknown"),
            ("compute_approval_id", ""),
            ("checkpoint_accuracy_matrix", [[0.5] * 4 for _ in range(4)]),
            ("checkpoint_balanced_accuracy_matrix", [[float("nan")] * 4 for _ in range(5)]),
        ):
            with self.subTest(key=key, value=str(value)[:20]):
                altered = copy.deepcopy(fake_record())
                altered[key] = value
                with self.assertRaises(ValueError):
                    validate_future_record(altered)
        altered = fake_record()
        del altered["compute_approval_id"]
        with self.assertRaises(ValueError):
            validate_future_record(altered)


if __name__ == "__main__":
    unittest.main()
