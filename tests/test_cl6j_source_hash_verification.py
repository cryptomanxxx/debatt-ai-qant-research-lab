"""CL-6J reads only four source files; never imports or executes models."""
from __future__ import annotations

import hashlib
import unittest

from integrations.continual_learning.cl6b_source_arm_equivalence import PATHS
from integrations.continual_learning.cl6j_source_hash_verification import (
    verify_checked_in_source_hashes,
)


def declared_current_sources() -> dict:
    # Commit and input digest are intentionally fabricated test placeholders.
    # This fixture is NOT evidence of a true Git commit or real dataset.
    return {
        "kind": "CL6I_DECLARED_PROVENANCE",
        "source_commit_sha": "a" * 40,
        "source_sha256": {
            name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in PATHS.items()
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


class CL6JSourceHashVerificationTests(unittest.TestCase):
    def test_current_source_bytes_match_but_git_and_data_do_not(self):
        result = verify_checked_in_source_hashes(declared_current_sources())
        self.assertEqual(result["decision"], "NO_GO")
        self.assertTrue(result["source_bytes_match_declaration"])
        self.assertEqual(result["source_names_checked"], sorted(PATHS))
        for key in (
            "source_commit_verified", "declared_input_set_verified",
            "paired_workload_verified", "training_executed",
            "model_executed", "physical_energy_measured",
            "scientific_equivalence_approved",
        ):
            self.assertIs(result[key], False, key)

    def test_mismatched_source_fingerprint_fails_closed(self):
        declaration = declared_current_sources()
        declaration["source_sha256"]["shared"] = "0" * 64
        with self.assertRaises(ValueError):
            verify_checked_in_source_hashes(declaration)

    def test_missing_source_name_or_bad_hash_fails_closed(self):
        declaration = declared_current_sources()
        del declaration["source_sha256"]["local"]
        with self.assertRaises(ValueError):
            verify_checked_in_source_hashes(declaration)
        declaration = declared_current_sources()
        declaration["source_sha256"]["dense"] = "bad"
        with self.assertRaises(ValueError):
            verify_checked_in_source_hashes(declaration)


if __name__ == "__main__":
    unittest.main()
