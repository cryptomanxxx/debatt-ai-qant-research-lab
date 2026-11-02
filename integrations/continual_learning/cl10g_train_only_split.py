"""CL-10G candidate train-only calibration split, with NO_GO execution status.

Pure algorithm for *supplied future training records*. Does not import/call
the dataset generator, construct models, train, infer or read heldout examples.
Tests use tiny hand-built schema fixtures, not real H1 data.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence

VERSION = "cl10g-train-only-stratified-hash-v1"
TASKS = ("T1", "T2", "T3", "T4")
SEEDS = (101, 102, 103, 104, 105)
FEATURES = 16
TRAIN_COUNT = 128
PER_CLASS = 64
VALID_PER_CLASS = 16


def _require(valid: bool, reason: str) -> None:
    if not valid:
        raise ValueError("CL-10G NO_GO: " + reason)


def _identity(features: object, label: object) -> str:
    _require(type(features) in (tuple, list) and len(features) == FEATURES,
             "invalid training feature shape")
    _require(all(type(bit) is int and bit in (0, 1) for bit in features),
             "invalid binary training features")
    _require(type(label) is int and label in (0, 1),
             "invalid training label")
    _require(label == (features[0] ^ features[1]), "global XOR label mismatch")
    return "".join(map(str, features))


def plan_train_only_calibration(
    seed: int, training_by_task: Mapping[str, Sequence[tuple[object, object]]]
) -> dict:
    """Returns only deterministic indices/digests, never a fitted model.

    Call with an already-authorized, frozen training partition in a later
    separately approved operation. Heldout data is not accepted or inspected.
    """
    _require(type(seed) is int and seed in SEEDS, "unplanned seed")
    _require(type(training_by_task) is dict and
             set(training_by_task) == set(TASKS), "exact four training tasks required")
    globally_seen = set()
    outputs = []
    for task in TASKS:
        rows = training_by_task[task]
        _require(type(rows) in (tuple, list) and len(rows) == TRAIN_COUNT,
                 "expected 128 original training records per task")
        by_class = {0: [], 1: []}
        identities = []
        for index, row in enumerate(rows):
            _require(type(row) in (tuple, list) and len(row) == 2,
                     "training record shape invalid")
            features, label = row
            bits = _identity(features, label)
            _require(bits not in globally_seen, "duplicate input across training tasks")
            globally_seen.add(bits)
            identities.append(bits)
            salt = f"{VERSION}|{seed}|{task}|{bits}".encode("ascii")
            rank = hashlib.sha256(salt).hexdigest()
            by_class[label].append((rank, bits, index))
        _require(all(len(by_class[c]) == PER_CLASS for c in (0, 1)),
                 "class quota drift from frozen H1 workload")
        validation = sorted(
            index for label in (0, 1)
            for _, _, index in sorted(by_class[label])[:VALID_PER_CLASS]
        )
        fit = sorted(set(range(TRAIN_COUNT)) - set(validation))
        _require(len(validation) == 32 and len(fit) == 96,
                 "calibration split size drift")
        # Binds the split to original ordered training content, not test data.
        source_digest = hashlib.sha256(
            json.dumps(identities, separators=(",", ":")).encode("ascii")
        ).hexdigest()
        partition_digest = hashlib.sha256(
            json.dumps({"fit": fit, "validation": validation},
                       sort_keys=True, separators=(",", ":")).encode("ascii")
        ).hexdigest()
        outputs.append({
            "task": task, "training_order_digest": source_digest,
            "fit_indices": fit, "validation_indices": validation,
            "partition_digest": partition_digest,
            "fit_per_class": 48, "validation_per_class": 16,
        })
    return {
        "kind": "CL10G_TRAIN_ONLY_SPLIT_CANDIDATE",
        "decision": "NO_GO",
        "seed": seed,
        "algorithm": VERSION,
        "tasks": outputs,
        "heldout_accessed": False,
        "training_authorized": False,
        "inference_authorized": False,
    }
