"""CL-3B record-shape specification: no training, result writes or data fetch."""
from __future__ import annotations

import math
from collections.abc import Sequence


def validate_result_shape(n_checkpoints: int, n_tasks: int) -> None:
    if (
        type(n_checkpoints) is not int or type(n_tasks) is not int
        or n_tasks < 2 or n_checkpoints != n_tasks + 1
    ):
        raise ValueError("expected n_tasks+1 checkpoints and at least 2 tasks")


def validate_accuracy_matrix(
    matrix: Sequence[Sequence[float]],
    *,
    n_tasks: int = 4,
    n_checkpoints: int = 5,
) -> None:
    """Pure shape/value validation for a FUTURE 5×4 held-out accuracy artifact."""
    validate_result_shape(n_checkpoints, n_tasks)
    if not isinstance(matrix, (list, tuple)) or len(matrix) != n_checkpoints:
        raise ValueError("incomplete checkpoint axis")
    for row in matrix:
        if not isinstance(row, (list, tuple)) or len(row) != n_tasks:
            raise ValueError("incomplete task axis")
        for value in row:
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("accuracy must be a finite fraction in [0, 1]")


def result_manifest_fields() -> tuple[str, ...]:
    """Mandatory provenance keys for any independently approved future runner."""
    return (
        "protocol_sha256",
        "source_commit_sha",
        "generator_source_sha256",
        "dataset_content_sha256",
        "python_version",
        "model_arm",
        "paired_seed",
        "training_update_count",
        "checkpoint_accuracy_matrix",
        "checkpoint_balanced_accuracy_matrix",
        "operation_scope_version",
        "scientific_review_id",
        "compute_approval_id",
    )
