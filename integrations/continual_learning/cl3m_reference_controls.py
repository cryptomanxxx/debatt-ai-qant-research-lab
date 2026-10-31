"""CL-3M rule-based dataset controls only; NEVER trains or scores an AI model.

Deterministic analytical references check the *candidate data definition*,
not model generalization or continual-learning performance.
"""
from __future__ import annotations

import argparse
import json

from integrations.continual_learning.cl2_paired_run_plan import load_candidate
from integrations.continual_learning.cl2_shared_rule_candidate import (
    generate_shared_rule_candidate, global_label,
)


def reference_partition(records: tuple) -> dict:
    """Compute exact fixed-class and XOR-oracle controls on proposed fixtures."""
    if not records:
        raise ValueError("empty candidate partition")
    labels = [record.label for record in records]
    if any(type(label) is not int or label not in (0, 1) for label in labels):
        raise ValueError("binary labels required")
    n_zero = labels.count(0)
    n_one = labels.count(1)
    if n_zero != n_one:
        raise ValueError("candidate balanced-label rule was violated")
    oracle_correct = sum(global_label(record.features) == record.label for record in records)
    if oracle_correct != len(records):
        raise ValueError("candidate shared XOR rule was violated")
    return {
        "count": len(records),
        "label_0_count": n_zero,
        "label_1_count": n_one,
        "always_predict_zero_accuracy": n_zero / len(records),
        "always_predict_one_accuracy": n_one / len(records),
        "known_xor_rule_accuracy": oracle_correct / len(records),
        "learned_model_accuracy": "NOT_MEASURED",
    }


def prepare_references() -> dict:
    """Generate only in-memory fixtures for the five fixed seeds."""
    plan = load_candidate()
    rows = []
    for seed in plan["paired_seeds"]:
        splits = generate_shared_rule_candidate(
            seed, train_per_task=plan["dataset"]["train_per_task"],
            test_per_task=plan["dataset"]["test_per_task"],
        )
        rows.append({
            "seed": seed,
            "tasks": [
                {
                    "task_id": split.task_id,
                    "train": reference_partition(split.train),
                    "test": reference_partition(split.test),
                }
                for split in splits
            ],
        })
    return {
        "kind": "CL3M_ANALYTICAL_DATA_CONTROLS_NOT_MODEL_RESULTS",
        "protocol_id": plan["protocol_id"],
        "paired_seeds": rows,
        "training_performed": False,
        "experiment_authorized": False,
        "physical_energy_measured": False,
        "note": "The perfect XOR oracle is built into the label generator; it is not evidence that a learned model can recover the rule.",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CL-3M synthetic rule controls, no model run")
    parser.add_argument("--check-data", action="store_true", required=True)
    args = parser.parse_args(argv)
    if args.check_data:
        print(json.dumps(prepare_references(), sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
