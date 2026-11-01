"""CL-6J read-only source-file fingerprint verifier; no model imports/runs.

Checks exactly four CL-2 Python source files at fixed repository paths.
Does not inspect or create experiment data, verify Git commit history, or train.
"""
from __future__ import annotations

import hashlib

from integrations.continual_learning.cl6b_source_arm_equivalence import PATHS
from integrations.continual_learning.cl6i_declared_provenance import (
    validate_declared_provenance,
)


def verify_checked_in_source_hashes(declaration: dict) -> dict:
    """Compare declared hashes with current on-disk source bytes, fail closed."""
    validate_declared_provenance(declaration)
    expected = declaration["source_sha256"]
    # PATHS is a fixed four-name mapping from CL-6B. Caller cannot select paths.
    if set(PATHS) != set(expected):
        raise ValueError("checked-in source inventory differs from declaration")
    verified = {}
    for name in sorted(PATHS):
        try:
            data = PATHS[name].read_bytes()
        except OSError as exc:
            raise ValueError(f"{name}: source file unavailable") from exc
        actual = hashlib.sha256(data).hexdigest()
        if actual != expected[name]:
            raise ValueError(f"{name}: source hash mismatch")
        verified[name] = actual
    return {
        "kind": "CL6J_SOURCE_BYTES_HASH_VERIFICATION",
        "decision": "NO_GO",
        "source_bytes_match_declaration": True,
        "source_names_checked": sorted(verified),
        "checked_source_sha256": verified,
        "source_commit_verified": False,
        "declared_input_set_verified": False,
        "paired_workload_verified": False,
        "training_executed": False,
        "model_executed": False,
        "physical_energy_measured": False,
        "scientific_equivalence_approved": False,
        "caveat": (
            "Only four current source files matched their declared SHA-256. "
            "Git commit authenticity, dependency tree, future input bytes, "
            "model state and experimental causal validity remain unverified."
        ),
    }
