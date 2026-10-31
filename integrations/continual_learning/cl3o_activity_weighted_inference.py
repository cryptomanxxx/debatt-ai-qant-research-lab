"""CL-3O synthetic-activity-weighted static prediction cost; NO model runs.

Combines known input-bit counts with CL-3N source-loop projections.
Generates only ephemeral synthetic data; does not call predict()/step().
"""
from __future__ import annotations

import argparse
import json

from integrations.continual_learning.cl2_data_candidate import INPUT_SIZE
from integrations.continual_learning.cl2_paired_run_plan import load_candidate
from integrations.continual_learning.cl2_shared_rule_candidate import (
    generate_shared_rule_candidate, shared_rule_dataset_sha256,
)
from integrations.continual_learning.cl3n_inference_accounting import (
    ARMS, INFERENCE_SCOPE_VERSION, estimate_prediction,
)


def summarize_partition_inputs(records: tuple) -> dict:
    """Sum per-example source-level estimates, not measured inference."""
    if not records:
        raise ValueError("empty partition")
    counts = {arm: {
        "forward_weight_terms": 0,
        "parameter_reads": 0,
        "parameter_writes": 0,
        "input_feature_visits": 0,
        "active_index_writes": 0,
        "active_index_reads": 0,
        "logical_payload_bytes": 0,
    } for arm in ARMS}
    total_active = 0
    for record in records:
        if len(record.features) != INPUT_SIZE or any(
            type(bit) is not int or bit not in (0, 1) for bit in record.features
        ):
            raise ValueError("expected exactly 16 binary input bits")
        active = sum(record.features)
        total_active += active
        for arm in ARMS:
            estimate = estimate_prediction(arm, active)
            metrics = counts[arm]
            for key in metrics:
                metrics[key] += getattr(estimate, key)
    return {
        "input_examples": len(records),
        "active_input_bits_total": total_active,
        "input_activity_fraction": total_active / (len(records) * INPUT_SIZE),
        "source_scope_version": INFERENCE_SCOPE_VERSION,
        "width_assumptions": {
            "parameter_scalar_bytes": 8,
            "input_scalar_bytes": 1,
            "active_index_bytes": 2,
        },
        "by_arm": counts,
        "training_or_predict_calls": 0,
        "physical_energy_joules": "NOT_MEASURED",
    }


def prepare_activity_weighted_preview() -> dict:
    plan = load_candidate()
    rows = []
    for seed in plan["paired_seeds"]:
        tasks = generate_shared_rule_candidate(
            seed, train_per_task=plan["dataset"]["train_per_task"],
            test_per_task=plan["dataset"]["test_per_task"],
        )
        rows.append({
            "seed": seed,
            "dataset_content_sha256": shared_rule_dataset_sha256(tasks),
            "tasks": [
                {
                    "task_id": task.task_id,
                    "train_input_preview": summarize_partition_inputs(task.train),
                    "heldout_input_preview": summarize_partition_inputs(task.test),
                }
                for task in tasks
            ],
        })
    return {
        "kind": "CL3O_STATIC_ACTIVITY_WEIGHTED_INFERENCE_PROXY_ONLY",
        "protocol_id": plan["protocol_id"],
        "paired_seeds": rows,
        "training_performed": False,
        "model_predictions_performed": False,
        "research_accuracy_measured": False,
        "physical_energy_measured": False,
        "note": "Heldout features are inspected only for static cost projection; heldout labels are not read for tuning or scoring. This is neither empirical model inference nor physical energy measurement.",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CL-3O static activity proxy; no model run")
    parser.add_argument("--preview", action="store_true", required=True)
    args = parser.parse_args(argv)
    if args.preview:
        print(json.dumps(prepare_activity_weighted_preview(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
