"""CL-3C deterministic synthetic-data provenance audit; NEVER trains a model.

Generates only the already-proposed in-memory synthetic fixtures to check
split leakage, label correctness, source identity and realized activity.
No network, hardware, model instantiation, output files, or training.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from integrations.continual_learning.cl2_data_candidate import ACTIVITY_RATES, INPUT_SIZE
from integrations.continual_learning.cl2_paired_run_plan import load_candidate
from integrations.continual_learning.cl2_shared_rule_candidate import (
    GLOBAL_LABEL_PAIR,
    generate_shared_rule_candidate,
    global_label,
    shared_rule_dataset_sha256,
)

GENERATOR_PATH = Path(__file__).with_name("cl2_shared_rule_candidate.py")


def audit_seed(seed: int, train_count: int, test_count: int) -> dict:
    """Audit records without returning features or feeding any model."""
    splits = generate_shared_rule_candidate(
        seed, train_per_task=train_count, test_per_task=test_count
    )
    unique_features: set[tuple[int, ...]] = set()
    tasks: list[dict] = []
    if [split.task_id for split in splits] != list(ACTIVITY_RATES):
        raise ValueError("unexpected task order")
    for split in splits:
        partitions = {}
        for name, records, count in (
            ("train", split.train, train_count),
            ("test", split.test, test_count),
        ):
            if len(records) != count:
                raise ValueError("incorrect partition size")
            positive = sum(record.label for record in records)
            if positive * 2 != count:
                raise ValueError("class imbalance")
            active = 0
            for record in records:
                if len(record.features) != INPUT_SIZE:
                    raise ValueError("incorrect input width")
                if any(type(bit) is not int or bit not in (0, 1) for bit in record.features):
                    raise ValueError("nonbinary input")
                if type(record.label) is not int or record.label != global_label(record.features):
                    raise ValueError("violated global XOR label")
                if record.features in unique_features:
                    raise ValueError("train/test or cross-task duplicate features")
                unique_features.add(record.features)
                active += sum(record.features)
            partitions[name] = {
                "count": count,
                "label_0": count - positive,
                "label_1": positive,
                "active_input_bits": active,
                "total_input_bits": count * INPUT_SIZE,
                "realized_activity_fraction": active / (count * INPUT_SIZE),
            }
        tasks.append({
            "task_id": split.task_id,
            "nominal_bernoulli_rate_pre_rejection": ACTIVITY_RATES[split.task_id],
            "train": partitions["train"],
            "test": partitions["test"],
        })
    return {
        "seed": seed,
        "dataset_content_sha256": shared_rule_dataset_sha256(splits),
        "unique_input_vectors": len(unique_features),
        "tasks": tasks,
    }


def prepare_audit() -> dict:
    """Produce audit-only metadata; leave CL-3A's NO_GO decision untouched."""
    plan = load_candidate()
    source_sha = hashlib.sha256(GENERATOR_PATH.read_bytes()).hexdigest()
    seeds = [
        audit_seed(seed, plan["dataset"]["train_per_task"], plan["dataset"]["test_per_task"])
        for seed in plan["paired_seeds"]
    ]
    digests = [item["dataset_content_sha256"] for item in seeds]
    if len(set(digests)) != len(digests):
        raise ValueError("paired seeds unexpectedly have identical datasets")
    return {
        "kind": "CL3C_SYNTHETIC_DATA_AUDIT_ONLY",
        "protocol_id": plan["protocol_id"],
        "generator_source_sha256": source_sha,
        "label_pair": list(GLOBAL_LABEL_PAIR),
        "paired_seeds": seeds,
        "training_performed": False,
        "experiment_authorized": False,
        "note": "Rejection sampling enforces balanced classes and changes nominal Bernoulli activity rates; this audit does not certify scientific construct validity.",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CL-3C in-memory synthetic data audit; no training")
    parser.add_argument("--audit", action="store_true", required=True)
    args = parser.parse_args(argv)
    if args.audit:
        print(json.dumps(prepare_audit(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
