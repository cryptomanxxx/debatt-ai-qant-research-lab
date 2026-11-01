"""CL-6I validate declared provenance metadata only; NO model/file execution.

A syntactically valid digest is not proof that referenced bytes exist or match.
This module never imports model implementations, opens datasets or runs training.
"""
from __future__ import annotations

import re
from collections.abc import Mapping


SOURCE_NAMES = ("dense", "local", "sparse", "shared")
PAIR_NAMES = ("primary_sparse", "secondary_index_reuse")
HEX40 = re.compile(r"[0-9a-f]{40}\Z")
HEX64 = re.compile(r"[0-9a-f]{64}\Z")


def validate_declared_provenance(record: Mapping[str, object]) -> dict:
    """Fail closed on incomplete claims; report schema validity, NOT verified provenance."""
    if not isinstance(record, Mapping):
        raise ValueError("provenance record must be a mapping")
    required = {
        "kind", "source_commit_sha", "source_sha256", "input_set_sha256",
        "pair_id", "paired_seed", "task_id", "sample_ordinal",
        "checkpoint_ordinal", "training_executed", "energy_measured",
    }
    if set(record) != required:
        raise ValueError("exact provenance field set required")
    if record["kind"] != "CL6I_DECLARED_PROVENANCE":
        raise ValueError("unsupported provenance kind")
    commit = record["source_commit_sha"]
    if not isinstance(commit, str) or HEX40.fullmatch(commit) is None:
        raise ValueError("full lowercase SHA-1 commit ID required")
    digests = record["source_sha256"]
    if not isinstance(digests, Mapping) or set(digests) != set(SOURCE_NAMES):
        raise ValueError("all four exact model source digests required")
    if any(not isinstance(d, str) or HEX64.fullmatch(d) is None for d in digests.values()):
        raise ValueError("invalid source SHA-256 digest")
    input_hash = record["input_set_sha256"]
    if not isinstance(input_hash, str) or HEX64.fullmatch(input_hash) is None:
        raise ValueError("input-set SHA-256 required")
    if record["pair_id"] not in PAIR_NAMES:
        raise ValueError("unregistered comparison")
    if type(record["paired_seed"]) is not int or record["paired_seed"] < 0:
        raise ValueError("nonnegative integer paired seed required")
    if record["task_id"] not in ("T1", "T2", "T3", "T4"):
        raise ValueError("unknown preregistered task")
    for key in ("sample_ordinal", "checkpoint_ordinal"):
        if type(record[key]) is not int or record[key] < 0:
            raise ValueError("nonnegative integer ordinal required")
    if record["training_executed"] is not False or record["energy_measured"] is not False:
        raise ValueError("not a read-only, unmeasured provenance declaration")
    return {
        "kind": "CL6I_PROVENANCE_SCHEMA_AUDIT",
        "decision": "NO_GO",
        "schema_valid": True,
        "referenced_bytes_verified": False,
        "dataset_existence_verified": False,
        "source_commit_verified": False,
        "model_executed": False,
        "training_executed": False,
        "physical_energy_measured": False,
        "caveat": "Format and completeness only. Digests are untrusted declarations until separately resolved and checked against immutable artifacts.",
    }
