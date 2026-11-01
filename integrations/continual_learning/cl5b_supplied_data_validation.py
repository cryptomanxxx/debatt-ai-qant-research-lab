"""CL-5B validation of *supplied* CL datasets; never calls a generator or trains.

The caller, not this module, is responsible for dataset generation authorization.
No scientific preregistration or NO_GO decision is modified by these checks.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json

from integrations.continual_learning.cl2_data_candidate import ACTIVITY_RATES, INPUT_SIZE
from integrations.continual_learning.cl2_shared_rule_candidate import global_label


@dataclass(frozen=True)
class SplitSummary:
    task_id: str
    partition: str
    count: int
    class_zero: int
    class_one: int
    mean_active_bits: float
    observed_activity_rate: float
    nominal_bernoulli_rate: float


def validate_supplied_splits(splits, *, train_per_task=128, test_per_task=64):
    """Fail closed on labels, shape, counts, class balance and cross-split leakage.

    Reports empirical activity without equating it to the nominal Bernoulli rate:
    rejection sampling conditioned on balance/uniqueness biases that distribution.
    This performs no training and never generates any examples.
    """
    if type(train_per_task) is not int or type(test_per_task) is not int:
        raise ValueError("partition sizes must be integers")
    if train_per_task <= 0 or test_per_task <= 0:
        raise ValueError("partition sizes must be positive")
    if len(splits) != len(ACTIVITY_RATES):
        raise ValueError("task count mismatch")
    if [s.task_id for s in splits] != list(ACTIVITY_RATES):
        raise ValueError("task order mismatch")
    used = set()
    rows = []
    for task in splits:
        for partition, expected in (("train", train_per_task), ("test", test_per_task)):
            records = getattr(task, partition)
            if len(records) != expected:
                raise ValueError(f"{task.task_id}/{partition}: record count mismatch")
            labels = Counter()
            active = 0
            for record in records:
                features = record.features
                if type(features) is not tuple or len(features) != INPUT_SIZE:
                    raise ValueError("invalid input shape")
                if any(type(value) is not int or value not in (0, 1) for value in features):
                    raise ValueError("nonbinary feature")
                if type(record.label) is not int or record.label not in (0, 1):
                    raise ValueError("invalid label")
                if record.label != global_label(features):
                    raise ValueError("global XOR label mismatch")
                if features in used:
                    raise ValueError("duplicate feature vector across tasks or partitions")
                used.add(features)
                labels[record.label] += 1
                active += sum(features)
            if expected % 2 or labels[0] != expected // 2 or labels[1] != expected // 2:
                raise ValueError(f"{task.task_id}/{partition}: class imbalance")
            rows.append(SplitSummary(
                task_id=task.task_id,
                partition=partition,
                count=expected,
                class_zero=labels[0],
                class_one=labels[1],
                mean_active_bits=active / expected,
                observed_activity_rate=active / (expected * INPUT_SIZE),
                nominal_bernoulli_rate=ACTIVITY_RATES[task.task_id],
            ))
    return {
        "kind": "CL5B_SUPPLIED_DATA_VALIDATION_ONLY",
        "decision": "NO_GO",
        "dataset_generation_executed_by_validator": False,
        "model_training_executed": False,
        "energy_measured": False,
        "records_checked": len(used),
        "partitions": [asdict(row) for row in rows],
    }


def compare_supplied_replicates(first, second, *, train_per_task=128, test_per_task=64):
    """Compare two independently supplied datasets, including record order.

    Callers must independently authorize and generate both inputs first. Matching
    records alone cannot prove reproducibility across processes or Python versions.
    """
    left = validate_supplied_splits(first, train_per_task=train_per_task, test_per_task=test_per_task)
    right = validate_supplied_splits(second, train_per_task=train_per_task, test_per_task=test_per_task)
    def canonical(splits):
        return [
            [task.task_id, [[list(r.features), r.label] for r in task.train],
             [[list(r.features), r.label] for r in task.test]]
            for task in splits
        ]
    def digest(value):
        encoded = json.dumps(canonical(value), separators=(",", ":"), ensure_ascii=True).encode("ascii")
        return hashlib.sha256(encoded).hexdigest()
    first_digest = digest(first)
    second_digest = digest(second)
    if first_digest != second_digest or canonical(first) != canonical(second):
        raise ValueError("supplied replicates differ")
    return {
        "kind": "CL5B_SUPPLIED_REPLICATE_COMPARISON_ONLY",
        "decision": "NO_GO",
        "matching": True,
        "canonical_record_sha256": first_digest,
        "records_checked_per_replicate": left["records_checked"],
        "dataset_generation_executed_by_validator": False,
        "model_training_executed": False,
        "energy_measured": False,
    }
