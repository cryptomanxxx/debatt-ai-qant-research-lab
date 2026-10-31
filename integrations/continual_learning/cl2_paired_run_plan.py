"""Structural validation only for a deliberately non-executable CL-2 plan.

The validator does NOT authorize compute, run a model, generate a dataset,
or alter project governance.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping


PLAN_PATH = Path(__file__).with_name("cl2_paired_run_plan_candidate.json")


def validate_candidate(plan: Mapping[str, object]) -> None:
    """Validate locked candidate constraints and reject execution claims."""
    if plan.get("status") != "draft_unapproved_unexecuted":
        raise ValueError("plan must remain an unapproved draft")
    for flag in ("scientific_review_approved", "compute_approved", "runner_enabled"):
        if plan.get(flag) is not False:
            raise ValueError(f"{flag} must be false")
    data = plan.get("dataset")
    if not isinstance(data, dict):
        raise ValueError("dataset section missing")
    if data.get("candidate") != "cl2-shared-rule-candidate-v0" or data.get("source_sha256") is not None:
        raise ValueError("generator must remain explicitly unpinned at draft stage")
    if (
        data.get("labels") != "xor_bits_0_1"
        or data.get("task_order") != ["T1", "T2", "T3", "T4"]
        or data.get("task_id_supplied_to_model") is not False
        or data.get("split_policy") != "globally_disjoint_within_seed"
        or data.get("train_per_task") != 128
        or data.get("test_per_task") != 64
    ):
        raise ValueError("task and split proposal changed")
    seeds = plan.get("paired_seeds")
    if seeds != [101, 102, 103, 104, 105]:
        raise ValueError("paired seed schedule changed")
    models = plan.get("models")
    if not isinstance(models, list) or len(models) != 2:
        raise ValueError("exactly two model arms required")
    if [m.get("arm") for m in models if isinstance(m, dict)] != ["dense_sgd", "local_hybrid"]:
        raise ValueError("unrecognized model arms")
    for model in models:
        if (
            model.get("parameters") != 154
            or [model.get("input_size"), model.get("hidden_size"), model.get("output_size")] != [16, 8, 2]
            or model.get("learning_rate") != 0.05
        ):
            raise ValueError("model budget or architecture changed")
    if models[0].get("name") != "TinyDenseMLP" or models[1].get("name") != "LocalHebbianMLP":
        raise ValueError("implementation changed")
    if models[1].get("local_rate") != 0.01:
        raise ValueError("local learning rate changed")
    if models[0].get("initial_seed_policy") != "same_seed_as_local" or models[1].get("initial_seed_policy") != "same_seed_as_dense":
        raise ValueError("paired initialization missing")
    training = plan.get("training")
    if not isinstance(training, dict) or (
        training.get("task_order_fixed") is not True
        or training.get("passes_per_task") != 1
        or training.get("observations_per_pass") != 128
        or training.get("updates_per_seed_ceiling") != 512
        or training.get("replay_enabled") is not False
        or training.get("hyperparameter_tuning_on_test") is not False
    ):
        raise ValueError("training schedule or leakage rule changed")
    evaluation = plan.get("evaluation")
    if not isinstance(evaluation, dict) or (
        evaluation.get("checkpoints") != ["before", "after_T1", "after_T2", "after_T3", "after_T4"]
        or evaluation.get("all_four_test_sets_each_checkpoint") is not True
        or evaluation.get("heldout_updates_allowed") is not False
        or evaluation.get("final_task_forgetting_defined") is not False
    ):
        raise ValueError("invalid checkpoint or holdout policy")
    gates = plan.get("interpretation_gates")
    if not isinstance(gates, dict) or (
        gates.get("majority_baseline_accuracy") != 0.5
        or gates.get("quality_floor_final_mean_accuracy") != 0.65
        or gates.get("paired_accuracy_tolerance_percentage_points") != 5
        or gates.get("energy_savings_claims_allowed") is not False
        or gates.get("operation_comparison_requires_same_scope_instrumentation") is not True
    ):
        raise ValueError("scientific inference gates changed")
    compute = plan.get("compute")
    if not isinstance(compute, dict) or (
        compute.get("external_requests_max") != 0
        or compute.get("hardware_calls_max") != 0
        or compute.get("cloud_budget_eur") != 0
        or compute.get("estimate_only") is not True
    ):
        raise ValueError("no external compute may be authorized by this draft")
    blockers = plan.get("review_blockers")
    expected = {
        "generator_source_sha256_missing", "scientific_review_pending",
        "compute_approval_pending", "training_runner_not_implemented",
        "matched_scope_instrumentation_missing",
    }
    if not isinstance(blockers, list) or set(blockers) != expected:
        raise ValueError("pre-execution blockers must remain explicit")


def load_candidate(path: Path = PLAN_PATH) -> dict:
    candidate = json.loads(path.read_text(encoding="utf-8"))
    validate_candidate(candidate)
    return candidate


if __name__ == "__main__":
    plan = load_candidate()
    print(f"{plan['protocol_id']}: structurally valid draft; NOT authorized or executed")
