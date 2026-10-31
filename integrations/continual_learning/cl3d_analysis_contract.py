"""CL-3D analysis rules for hypothetical future held-out results.

Pure, deterministic arithmetic. No model, dataset, training, I/O or energy
measurement. Test fixtures MUST NOT be reported as experimental findings.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from integrations.continual_learning.cl3b_result_contract import validate_accuracy_matrix

SEEDS = (101, 102, 103, 104, 105)
TASK_COUNT = 4
MAJORITY_BASELINE = 0.5
QUALITY_FLOOR = 0.65
MATCHED_TOLERANCE = 0.05


def summarize_accuracy(matrix: Sequence[Sequence[float]]) -> dict:
    """All tasks evaluated at all checkpoints; final task has no forgetting score.

    For task t, use checkpoints after its training but BEFORE the final row.
    Forgetting = best earlier post-training accuracy - final accuracy.
    Negative forgetting records genuine later improvement, per CL-2 protocol.
    """
    validate_accuracy_matrix(matrix, n_tasks=TASK_COUNT, n_checkpoints=TASK_COUNT + 1)
    final = list(matrix[-1])
    forgetting: list[float | None] = []
    for task_idx in range(TASK_COUNT):
        if task_idx == TASK_COUNT - 1:
            forgetting.append(None)
        else:
            forgetting.append(max(row[task_idx] for row in matrix[task_idx + 1:TASK_COUNT]) - final[task_idx])
    return {
        "final_accuracy_by_task": final,
        "final_mean_accuracy": sum(final) / TASK_COUNT,
        "forgetting_by_task": forgetting,
        "mean_forgetting_prior_tasks": sum(x for x in forgetting if x is not None) / (TASK_COUNT - 1),
        "majority_baseline": MAJORITY_BASELINE,
    }


def paired_summary(
    dense: Mapping[int, Sequence[Sequence[float]]],
    local: Mapping[int, Sequence[Sequence[float]]],
) -> dict:
    """Descriptive, paired-seed arithmetic only; no p-values or energy claims."""
    if set(dense) != set(SEEDS) or set(local) != set(SEEDS):
        raise ValueError("exact paired seed schedule required")
    if any(type(seed) is not int for seed in (*dense.keys(), *local.keys())):
        raise ValueError("seed keys must be integers")
    rows = []
    for seed in SEEDS:
        a, b = summarize_accuracy(dense[seed]), summarize_accuracy(local[seed])
        delta = b["final_mean_accuracy"] - a["final_mean_accuracy"]
        rows.append({
            "seed": seed,
            "dense_final_accuracy": a["final_mean_accuracy"],
            "local_final_accuracy": b["final_mean_accuracy"],
            "local_minus_dense_accuracy": delta,
            "dense_prior_task_mean_forgetting": a["mean_forgetting_prior_tasks"],
            "local_prior_task_mean_forgetting": b["mean_forgetting_prior_tasks"],
        })
    differences = [x["local_minus_dense_accuracy"] for x in rows]
    mean_delta = sum(differences) / len(differences)
    paired_sd = math.sqrt(sum((x - mean_delta) ** 2 for x in differences) / (len(differences) - 1))
    quality_pass = all(
        (x["dense_final_accuracy"] >= QUALITY_FLOOR or math.isclose(x["dense_final_accuracy"], QUALITY_FLOOR, rel_tol=0.0, abs_tol=1e-12))
        and (x["local_final_accuracy"] >= QUALITY_FLOOR or math.isclose(x["local_final_accuracy"], QUALITY_FLOOR, rel_tol=0.0, abs_tol=1e-12))
        for x in rows
    )
    parity_pass = all(
        abs(x) <= MATCHED_TOLERANCE
        or math.isclose(abs(x), MATCHED_TOLERANCE, rel_tol=0.0, abs_tol=1e-12)
        for x in differences
    )
    return {
        "kind": "DESCRIPTIVE_PAIRED_SUMMARY_ONLY",
        "paired_seeds": rows,
        "mean_local_minus_dense_accuracy": mean_delta,
        "paired_sample_sd": paired_sd,
        "quality_floor": QUALITY_FLOOR,
        "matched_accuracy_tolerance": MATCHED_TOLERANCE,
        "quality_gate_all_seeds": quality_pass,
        "parity_gate_all_seeds": parity_pass,
        "logical_cost_comparison_interpretable": quality_pass and parity_pass,
        "physical_energy_inference_allowed": False,
        "statistical_significance_inference_allowed": False,
    }
