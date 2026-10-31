"""CL-3J future experiment artifact validation only; NO execution or writes.

A structurally valid result is NEVER proof of an approved or executed run.
This module does not load actual runs, train, measure, infer or save outputs.
"""
from __future__ import annotations

import re
from collections.abc import Mapping

from integrations.continual_learning.cl3b_result_contract import (
    result_manifest_fields, validate_accuracy_matrix,
)
from integrations.continual_learning.cl3d_analysis_contract import SEEDS
from integrations.continual_learning.cl3e_measurement_ledger import SCOPE_VERSION

ARMS = ("dense_sgd", "local_hybrid")
REQUIRED = frozenset(result_manifest_fields())
RE_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
RE_GIT_COMMIT = re.compile(r"[0-9a-f]{40}\Z")


def validate_future_record(record: Mapping[str, object]) -> None:
    """Validate structure, not the authenticity of evidence or approvals.

    A future runner must separately verify every ID/digest against actual
    immutable artifacts and signed human/scientific decisions.
    """
    if not isinstance(record, dict) or set(record) != REQUIRED:
        raise ValueError("future manifest fields must match the declared contract")
    for field in ("protocol_sha256", "generator_source_sha256", "dataset_content_sha256"):
        value = record[field]
        if not isinstance(value, str) or not RE_SHA256.fullmatch(value):
            raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    source = record["source_commit_sha"]
    if not isinstance(source, str) or not RE_GIT_COMMIT.fullmatch(source):
        raise ValueError("source_commit_sha must be a full 40-hex commit")
    if record["model_arm"] not in ARMS:
        raise ValueError("unknown model arm")
    if type(record["paired_seed"]) is not int or record["paired_seed"] not in SEEDS:
        raise ValueError("unexpected paired seed")
    if type(record["training_update_count"]) is not int or not 0 <= record["training_update_count"] <= 512:
        raise ValueError("outside proposed per-seed/arm update cap")
    if record["operation_scope_version"] != SCOPE_VERSION:
        raise ValueError("wrong operation scope version")
    if not isinstance(record["python_version"], str) or not re.fullmatch(r"3\.\d+\.\d+", record["python_version"]):
        raise ValueError("Python patch version required")
    for field in ("scientific_review_id", "compute_approval_id"):
        if not isinstance(record[field], str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{5,127}", record[field]):
            raise ValueError(f"{field} must have a traceable identifier")
    for field in ("checkpoint_accuracy_matrix", "checkpoint_balanced_accuracy_matrix"):
        validate_accuracy_matrix(record[field])


def artifact_review_status() -> dict:
    """Pre-experiment readiness only; no result instances created."""
    return {
        "kind": "CL3J_FUTURE_ARTIFACT_SCHEMA_ONLY",
        "required_fields": sorted(REQUIRED),
        "paired_seeds": list(SEEDS),
        "allowed_arms": list(ARMS),
        "training_allowed": False,
        "physical_energy_inference_allowed": False,
        "schema_validation_proves_authenticity": False,
        "missing": [
            "independent_scientific_signoff",
            "separate_explicit_human_compute_approval",
            "approved_runner_and_immutable_source_hash",
            "actual_verified_result_artifacts",
        ],
    }
