"""CL-12: inspect externally supplied H1 records, never generate any records.

The digest is a CONTENT checksum, not an authenticity proof or approval to
access/generate data. No model, generator, network, or subprocess imports.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from cl10g_train_only_split import plan_train_only_calibration

TASKS = ("T1", "T2", "T3", "T4")
SEEDS = (101, 102, 103, 104, 105)


def _reject(reason: str) -> None:
    raise ValueError("CL-12 NO_GO: " + reason)


def inspect_supplied_seed_records(seed: int, supplied: object) -> dict:
    """Validate ordered supplied data; digest matches CL-2 canonical encoding.

    Input is a list of 4 {"task_id", "train", "test"} objects containing
    [16 binary features, XOR label] rows. Never calls a generator.
    """
    if type(seed) is not int or seed not in SEEDS:
        _reject("unplanned seed")
    if type(supplied) is not list or len(supplied) != 4:
        _reject("expected four task objects")
    canonical = []
    training_by_task = {}
    seen = set()
    for task_id, task in zip(TASKS, supplied):
        if type(task) is not dict or set(task) != {"task_id", "train", "test"}:
            _reject("invalid task schema")
        if task["task_id"] != task_id:
            _reject("task order mismatch")
        normalized = {"task_id": task_id}
        for partition, required in (("train", 128), ("test", 64)):
            records = task[partition]
            if type(records) is not list or len(records) != required:
                _reject("partition size mismatch")
            normalized_rows = []
            for row in records:
                if type(row) not in (list, tuple) or len(row) != 2:
                    _reject("invalid row shape")
                features, label = row
                if (type(features) not in (list, tuple) or len(features) != 16
                        or any(type(bit) is not int or bit not in (0, 1)
                               for bit in features)
                        or type(label) is not int or label not in (0, 1)):
                    _reject("non-binary or ill-shaped data")
                if label != (features[0] ^ features[1]):
                    _reject("XOR label mismatch")
                identity = tuple(features)
                if identity in seen:
                    _reject("duplicate features across tasks or partitions")
                seen.add(identity)
                normalized_rows.append([list(features), label])
            if Counter(row[1] for row in normalized_rows) != {0: required // 2, 1: required // 2}:
                _reject("class balance mismatch")
            normalized[partition] = normalized_rows
        training_by_task[task_id] = [tuple(row) for row in normalized["train"]]
        canonical.append(normalized)

    encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=True).encode("ascii")
    split = plan_train_only_calibration(seed, training_by_task)
    return {
        "decision": "NO_GO",
        "scope": "SUPPLIED_RECORD_VALIDATION_ONLY",
        "seed": seed,
        "record_count": len(seen),
        "train_count": 512,
        "heldout_count": 256,
        "dataset_sha256_candidate": hashlib.sha256(encoded).hexdigest(),
        "train_only_partition_sha256_by_task": {
            task["task"]: task["partition_digest"] for task in split["tasks"]
        },
        "data_origin_authenticated": False,
        "generator_executed": False,
        "dataset_generation_authorized": False,
        "training_authorized": False,
        "inference_authorized": False,
        "physical_energy": "NOT_MEASURED",
    }


def inspect_existing_json(path: Path, seed: int) -> dict:
    """Read only an existing bounded regular file; caller must authorize access."""
    path = Path(path)
    if not path.is_file() or path.is_symlink() or path.stat().st_size > 200_000:
        _reject("missing, linked or oversized supplied file")
    return inspect_supplied_seed_records(
        seed, json.loads(path.read_text(encoding="ascii"))
    )
