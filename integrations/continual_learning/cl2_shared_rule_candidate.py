"""Alternative primary CL-2 dataset candidate with one global label function.

This module only constructs in-memory examples; it never trains a model,
writes files, contacts hardware or modifies the unapproved CL-2 protocol.
"""
from __future__ import annotations

import hashlib
import random

from integrations.continual_learning.cl2_data_candidate import (
    ACTIVITY_RATES,
    INPUT_SIZE,
    Example,
    TaskSplit,
    dataset_sha256,
)


GLOBAL_LABEL_PAIR = (0, 1)
VERSION = "cl2-shared-rule-candidate-v0"


def global_label(features: tuple[int, ...]) -> int:
    if len(features) != INPUT_SIZE or any(type(v) is not int or v not in (0, 1) for v in features):
        raise ValueError("expected 16 binary input values")
    return features[GLOBAL_LABEL_PAIR[0]] ^ features[GLOBAL_LABEL_PAIR[1]]


def generate_shared_rule_candidate(
    seed: int, *, train_per_task: int = 128, test_per_task: int = 64
) -> tuple[TaskSplit, ...]:
    """Four activity-shifted tasks, identical global XOR rule, no task-ID input.

    Rejection sampling balances classes and enforces globally distinct input
    vectors for this seed. It changes the natural Bernoulli distribution and
    must not be interpreted as an unbiased IID sample from it.
    """
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if any(type(n) is not int or n <= 0 or n % 2 for n in (train_per_task, test_per_task)):
        raise ValueError("train/test counts must be positive and even")
    if len(ACTIVITY_RATES) * (train_per_task + test_per_task) > 2 ** INPUT_SIZE:
        raise ValueError("input space too small for globally unique records")
    used: set[tuple[int, ...]] = set()
    tasks: list[TaskSplit] = []
    for task_id, probability in ACTIVITY_RATES.items():
        material = hashlib.sha256(f"{VERSION}|{seed}|{task_id}".encode("ascii")).digest()
        rng = random.Random(int.from_bytes(material, "big"))
        partitions: list[tuple[Example, ...]] = []
        for count in (train_per_task, test_per_task):
            remaining = [count // 2, count // 2]
            records: list[Example] = []
            for _ in range(1_000_000):
                if not any(remaining):
                    break
                features = tuple(int(rng.random() < probability) for _ in range(INPUT_SIZE))
                label = global_label(features)
                if remaining[label] == 0 or features in used:
                    continue
                used.add(features)
                records.append(Example(features, label))
                remaining[label] -= 1
            if any(remaining):
                raise ValueError("insufficient unique class-balanced records")
            partitions.append(tuple(records))
        tasks.append(TaskSplit(task_id, partitions[0], partitions[1]))
    return tuple(tasks)


def shared_rule_dataset_sha256(splits: tuple[TaskSplit, ...]) -> str:
    """Reuse existing canonical record digest; task design/version pinned separately."""
    return dataset_sha256(splits)
