"""CL-6I literal metadata fixture tests: no model or dataset access."""
from __future__ import annotations

import unittest

from integrations.continual_learning.cl6i_declared_provenance import (
    validate_declared_provenance,
)


def declaration() -> dict:
    # Test-only placeholder hex strings, NOT real commit, source or data hashes.
    return {
        "kind": "CL6I_DECLARED_PROVENANCE",
        "source_commit_sha": "a" * 40,
        "source_sha256": {
            "dense": "b" * 64,
            "local": "c" * 64,
            "sparse": "d" * 64,
            "shared": "e" * 64,
        },
        "input_set_sha256": "f" * 64,
        "pair_id": "primary_sparse",
        "paired_seed": 101,
        "task_id": "T1",
        "sample_ordinal": 0,
        "checkpoint_ordinal": 0,
        "training_executed": False,
        "energy_measured": False,
    }


class CL6IDeclaredProvenanceTests(unittest.TestCase):
    def test_schema_only_never_verifies_claims(self):
        result = validate_declared_provenance(declaration())
        self.assertEqual(result["decision"], "NO_GO")
        self.assertTrue(result["schema_valid"])
        for key in (
            "referenced_bytes_verified", "dataset_existence_verified",
            "source_commit_verified", "model_executed",
            "training_executed", "physical_energy_measured",
        ):
            self.assertIs(result[key], False, key)

    def test_missing_and_unrecognized_fields_fail(self):
        for modification in ("missing", "extra"):
            record = declaration()
            if modification == "missing":
                del record["input_set_sha256"]
            else:
                record["verified"] = True
            with self.assertRaises(ValueError):
                validate_declared_provenance(record)

    def test_bad_hash_shapes_and_missing_sources_fail(self):
        for key, value in (
            ("source_commit_sha", "a" * 39),
            ("input_set_sha256", "Z" * 64),
            ("source_commit_sha", "A" * 40),
        ):
            record = declaration()
            record[key] = value
            with self.assertRaises(ValueError):
                validate_declared_provenance(record)
        record = declaration()
        del record["source_sha256"]["shared"]
        with self.assertRaises(ValueError):
            validate_declared_provenance(record)

    def test_approval_and_identity_inconsistencies_fail(self):
        for key, value in (
            ("training_executed", True),
            ("energy_measured", True),
            ("pair_id", "dense_vs_local"),
            ("paired_seed", True),
            ("task_id", "T5"),
            ("sample_ordinal", -1),
        ):
            record = declaration()
            record[key] = value
            with self.assertRaises(ValueError):
                validate_declared_provenance(record)


if __name__ == "__main__":
    unittest.main()
