"""CL-3P source-level *hypothetical* training-access proxy; NO training.

Uses synthetic training input activity only; never instantiates a model,
calls step(), touches heldout labels or reports real memory/energy usage.
"""
from __future__ import annotations

import argparse
import json

from integrations.continual_learning.cl2_data_candidate import INPUT_SIZE
from integrations.continual_learning.cl2_operation_accounting import estimate_step
from integrations.continual_learning.cl2_paired_run_plan import load_candidate
from integrations.continual_learning.cl2_payload_proxy import estimate_payload_proxy
from integrations.continual_learning.cl2_shared_rule_candidate import generate_shared_rule_candidate

ARMS = ("dense_sgd", "local_hybrid", "sparse_local_hybrid", "sparse_shared_index")
SOURCE_SCOPE_VERSION = "cl2-operation-accounting-v0-training-proxy"


def summarize_training_inputs(training_records: tuple) -> dict:
    """Use *one assumed update per training example* for static arithmetic."""
    if not training_records or len(training_records) > 128:
        raise ValueError("expected bounded training input partition")
    totals = {arm: {
        "forward_weight_terms": 0,
        "parameter_reads": 0,
        "parameter_writes": 0,
        "input_feature_checks": 0,
        "logical_payload_bytes": 0,
    } for arm in ARMS}
    active_sum = 0
    for record in training_records:
        features = record.features
        if len(features) != INPUT_SIZE or any(
            type(bit) is not int or bit not in (0, 1) for bit in features
        ):
            raise ValueError("expected sixteen binary input bits")
        active = sum(features)
        active_sum += active
        for arm in ARMS:
            step = estimate_step(arm, active)
            payload = estimate_payload_proxy(arm, active)
            row = totals[arm]
            row["forward_weight_terms"] += step.forward_weight_terms
            row["parameter_reads"] += step.total_parameter_reads
            row["parameter_writes"] += step.total_parameter_writes
            row["input_feature_checks"] += step.input_feature_checks
            row["logical_payload_bytes"] += payload.total_logical_payload_bytes
    return {
        "hypothetical_update_count": len(training_records),
        "active_input_bits_total": active_sum,
        "input_activity_fraction": active_sum / (len(training_records) * INPUT_SIZE),
        "scope_version": SOURCE_SCOPE_VERSION,
        "assumed_bytes": {"parameter": 8, "input": 1, "active_index": 2},
        "by_arm": totals,
        "model_updates_performed": 0,
        "physical_energy_joules": "NOT_MEASURED",
    }


def prepare_training_proxy() -> dict:
    """In-memory synthetic features only, locked draft update budget."""
    plan = load_candidate()
    seeds = []
    for seed in plan["paired_seeds"]:
        splits = generate_shared_rule_candidate(
            seed, train_per_task=plan["dataset"]["train_per_task"],
            test_per_task=plan["dataset"]["test_per_task"],
        )
        tasks = [
            {"task_id": task.task_id, "training_input_proxy": summarize_training_inputs(task.train)}
            for task in splits
        ]
        total_updates = sum(task["training_input_proxy"]["hypothetical_update_count"] for task in tasks)
        if total_updates != plan["training"]["updates_per_seed_ceiling"]:
            raise ValueError("proposed update count drifted")
        seeds.append({"seed": seed, "hypothetical_updates_per_arm": total_updates, "tasks": tasks})
    return {
        "kind": "CL3P_STATIC_ACTIVITY_WEIGHTED_TRAINING_PROXY_ONLY",
        "protocol_id": plan["protocol_id"],
        "paired_seeds": seeds,
        "training_performed": False,
        "heldout_labels_accessed": False,
        "physical_energy_measured": False,
        "note": "Per-example updates are assumed for source-loop accounting, not executed. Only predeclared training input features determine counts.",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CL-3P synthetic training-input cost projection only")
    parser.add_argument("--preview", action="store_true", required=True)
    args = parser.parse_args(argv)
    if args.preview:
        print(json.dumps(prepare_training_proxy(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
