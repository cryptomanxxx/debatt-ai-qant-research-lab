"""Fictional matrices/digests only. Never runs experiment training."""
import copy
import unittest

from integrations.continual_learning.cl3d_analysis_contract import SEEDS
from integrations.continual_learning.cl3j_artifact_contract import ARMS
from integrations.continual_learning.cl3k_pair_integrity import validate_paired_manifest_set


def fixture_records():
    records = []
    for seed_index, seed in enumerate(SEEDS):
        for arm in ARMS:
            records.append({
                "protocol_sha256": "a" * 64,
                "source_commit_sha": "b" * 40,
                "generator_source_sha256": "c" * 64,
                "dataset_content_sha256": format(seed_index + 1, "064x"),
                "python_version": "3.11.9",
                "model_arm": arm,
                "paired_seed": seed,
                "training_update_count": 512,
                "checkpoint_accuracy_matrix": [[0.5] * 4 for _ in range(5)],
                "checkpoint_balanced_accuracy_matrix": [[0.5] * 4 for _ in range(5)],
                "operation_scope_version": "cl3e-source-loop-estimate-v0",
                "scientific_review_id": "FAKE-scientific-review",
                "compute_approval_id": "FAKE-compute-approval",
            })
    return records


class CL3KPairIntegrityContracts(unittest.TestCase):
    def test_full_fictional_set_has_required_pairing(self):
        summary = validate_paired_manifest_set(fixture_records())
        self.assertTrue(summary["pairs_complete"])
        self.assertFalse(summary["training_allowed"])
        self.assertFalse(summary["provenance_authenticated"])
        self.assertFalse(summary["approval_ids_authenticated"])

    def test_reject_missing_duplicate_and_mismatched_metadata(self):
        records = fixture_records()
        with self.assertRaises(ValueError):
            validate_paired_manifest_set(records[:-1])
        wrong = copy.deepcopy(records)
        wrong[0] = copy.deepcopy(wrong[1])
        with self.assertRaises(ValueError):
            validate_paired_manifest_set(wrong)
        for key, value in (
            ("dataset_content_sha256", "f" * 64),
            ("protocol_sha256", "e" * 64),
            ("source_commit_sha", "f" * 40),
            ("compute_approval_id", "FAKE-other-approval"),
            ("training_update_count", 511),
        ):
            wrong = copy.deepcopy(records)
            wrong[1][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_paired_manifest_set(wrong)

    def test_reject_reused_dataset_digest_across_distinct_seeds(self):
        records = fixture_records()
        for row in records[2:4]:
            row["dataset_content_sha256"] = records[0]["dataset_content_sha256"]
        with self.assertRaises(ValueError):
            validate_paired_manifest_set(records)


if __name__ == "__main__":
    unittest.main()
