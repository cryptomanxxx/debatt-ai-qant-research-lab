"""Candidate CL-2 binary dataset construction (NOT an approved experiment).

Pure standard-library fixture generation, no data downloads, model fitting,
hardware access, files written or experiment execution on import.

This is a candidate for scientific protocol review. It deliberately does not
modify cl2_protocol_draft.json, whose generator remains unimplemented.
"""
from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass


INPUT_SIZE = 16
TASK_PAIRS = {"T1": (0, 1), "T2": (2, 3), "T3": (4, 5), "T4": (6, 7)}
ACTIVITY_RATES = {"T1": 0.125, "T2": 0.25, "T3": 0.375, "T4": 0.5}


@dataclass(frozen=True)
class Example:
    features: tuple[int, ...]
    label: int


@dataclass(frozen=True)
class TaskSplit:
    task_id: str
    train: tuple[Example, ...]
    test: tuple[Example, ...]


def _rng(seed: int, task_id: str) -> random.Random:
    digest = hashlib.sha256(f"cl2-candidate-v0|{seed}|{task_id}".encode("ascii")).digest()
    return random.Random(int.from_bytes(digest, "big"))


def generate_candidate(
    seed: int, *, train_per_task: int = 128, test_per_task: int = 64
) -> tuple[TaskSplit, ...]:
    """Create class-balanced disjoint binary inputs across all tasks and splits.

    Labels are task-specific XOR rules. The same feature vector could have
    conflicting labels under different tasks; we prevent identical vectors
    across tasks, but this does NOT remove the underlying conceptual conflict.
    Class balancing by rejection changes the natural Bernoulli distribution.
    """
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if any(type(n) is not int or n <= 0 or n % 2 for n in (train_per_task, test_per_task)):
        raise ValueError("positive even train and test sizes required")
    if 4 * (train_per_task + test_per_task) > 2**INPUT_SIZE:
        raise ValueError("requested more unique inputs than input space")

    used: set[tuple[int, ...]] = set()
    result: list[TaskSplit] = []
    for task_id, pair in TASK_PAIRS.items():
        rng = _rng(seed, task_id)
        rate = ACTIVITY_RATES[task_id]
        allocations = (train_per_task, test_per_task)
        partitions: list[tuple[Example, ...]] = []
        for count in allocations:
            targets = [count // 2, count // 2]
            rows: list[Example] = []
            for _ in range(1_000_000):
                if not any(targets):
                    break
                features = tuple(int(rng.random() < rate) for _ in range(INPUT_SIZE))
                label = features[pair[0]] ^ features[pair[1]]
                if targets[label] <= 0 or features in used:
                    continue
                used.add(features)
                rows.append(Example(features, label))
                targets[label] -= 1
            if any(targets):
                raise ValueError("insufficient unique balanced samples under candidate sampler")
            partitions.append(tuple(rows))
        result.append(TaskSplit(task_id, partitions[0], partitions[1]))
    return tuple(result)


def dataset_sha256(splits: tuple[TaskSplit, ...]) -> str:
    """Stable content digest for proposed example ordering, labels and splits."""
    payload = [
        {
            "task_id": split.task_id,
            "train": [[list(e.features), e.label] for e in split.train],
            "test": [[list(e.features), e.label] for e in split.test],
        }
        for split in splits
    ]
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("ascii")).hexdigest()
