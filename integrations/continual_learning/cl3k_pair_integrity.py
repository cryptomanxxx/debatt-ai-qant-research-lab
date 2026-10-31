"""CL-3K pure cross-record parity validation; NO run, I/O or training.

Requires two arms per seed with identical declared provenance. Passing
format checks does not authenticate approval IDs or dataset hashes.
"""
from __future__ import annotations

from collections.abc import Sequence

from integrations.continual_learning.cl3d_analysis_contract import SEEDS
from integrations.continual_learning.cl3j_artifact_contract import (
    ARMS, validate_future_record,
)

SHARED_FIELDS = (
    "protocol_sha256", "source_commit_sha", "generator_source_sha256",
    "python_version", "operation_scope_version", "scientific_review_id",
    "compute_approval_id",
)


def validate_paired_manifest_set(records: Sequence[dict]) -> dict:
    """Reject missing arms, mismatched provenance or repeat seed records."""
    if not isinstance(records, (list, tuple)) or len(records) != len(SEEDS) * len(ARMS):
        raise ValueError("exactly ten future records required")
    indexed: dict[tuple[int, str], dict] = {}
    for record in records:
        validate_future_record(record)
        key = (record["paired_seed"], record["model_arm"])
        if key in indexed:
            raise ValueError("duplicate seed-arm record")
        indexed[key] = record
    required = {(seed, arm) for seed in SEEDS for arm in ARMS}
    if set(indexed) != required:
        raise ValueError("missing or unexpected paired seed/arm")
    reference = indexed[(SEEDS[0], ARMS[0])]
    if any(
        record[field] != reference[field]
        for record in records for field in SHARED_FIELDS
    ):
        raise ValueError("global protocol/software/approval metadata mismatch")
    seed_dataset_hashes = []
    for seed in SEEDS:
        pair = [indexed[(seed, arm)] for arm in ARMS]
        if pair[0]["dataset_content_sha256"] != pair[1]["dataset_content_sha256"]:
            raise ValueError("paired arms must use exactly the same seed dataset")
        if pair[0]["training_update_count"] != pair[1]["training_update_count"]:
            raise ValueError("paired arms must have the same update budget")
        seed_dataset_hashes.append(pair[0]["dataset_content_sha256"])
    if len(set(seed_dataset_hashes)) != len(SEEDS):
        raise ValueError("different seeds unexpectedly share a dataset digest")
    return {
        "kind": "CL3K_PAIRED_MANIFEST_SHAPE_ONLY",
        "seeds": list(SEEDS),
        "arms": list(ARMS),
        "pairs_complete": True,
        "shared_metadata_fields_checked": list(SHARED_FIELDS),
        "training_allowed": False,
        "provenance_authenticated": False,
        "approval_ids_authenticated": False,
        "energy_inference_allowed": False,
    }
