"""CL-3G class-conditional synthetic data audit; NO training or model inference.

Uses the existing candidate generator in memory and returns only aggregated
data-distribution diagnostics. This is scientific protocol review input.
"""
from __future__ import annotations

import argparse
import json

from integrations.continual_learning.cl2_data_candidate import INPUT_SIZE
from integrations.continual_learning.cl2_paired_run_plan import load_candidate
from integrations.continual_learning.cl2_shared_rule_candidate import (
    GLOBAL_LABEL_PAIR, generate_shared_rule_candidate,
)


def conditional_partition(records: tuple) -> dict:
    """Summarize activity separately for each binary class; fail on imbalance."""
    if not records:
        raise ValueError("cannot audit an empty partition")
    classes: dict[str, dict] = {}
    for label in (0, 1):
        selected = [record for record in records if record.label == label]
        if not selected:
            raise ValueError("missing class")
        if len(selected) * 2 != len(records):
            raise ValueError("expected strict balanced class split")
        active = sum(sum(record.features) for record in selected)
        label_pair_active = sum(
            sum(record.features[index] for index in GLOBAL_LABEL_PAIR)
            for record in selected
        )
        classes[str(label)] = {
            "count": len(selected),
            "active_input_bits": active,
            "activity_fraction": active / (len(selected) * INPUT_SIZE),
            "label_pair_active_bits": label_pair_active,
            "label_pair_activity_fraction": label_pair_active / (len(selected) * 2),
        }
    return {
        "by_label": classes,
        "absolute_conditional_activity_gap": abs(
            classes["1"]["activity_fraction"] - classes["0"]["activity_fraction"]
        ),
    }


def audit_conditional_seed(seed: int, train_per_task: int = 128, test_per_task: int = 64) -> dict:
    """Read-only, aggregated observations on in-memory synthetic examples."""
    tasks = generate_shared_rule_candidate(
        seed, train_per_task=train_per_task, test_per_task=test_per_task
    )
    return {
        "seed": seed,
        "tasks": [
            {
                "task_id": task.task_id,
                "train": conditional_partition(task.train),
                "test": conditional_partition(task.test),
            }
            for task in tasks
        ],
    }


def prepare_conditional_audit() -> dict:
    plan = load_candidate()
    return {
        "kind": "CL3G_SYNTHETIC_CONDITIONAL_AUDIT_ONLY",
        "protocol_id": plan["protocol_id"],
        "paired_seeds": [
            audit_conditional_seed(
                seed, plan["dataset"]["train_per_task"], plan["dataset"]["test_per_task"]
            )
            for seed in plan["paired_seeds"]
        ],
        "training_performed": False,
        "scientific_review_approved": False,
        "experiment_authorized": False,
        "note": "Class balancing and rejection sampling distort raw Bernoulli activity; these diagnostics are not model accuracy, energy or proof of task validity.",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CL-3G synthetic conditional audit; no training")
    parser.add_argument("--audit", action="store_true", required=True)
    args = parser.parse_args(argv)
    if args.audit:
        print(json.dumps(prepare_conditional_audit(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
