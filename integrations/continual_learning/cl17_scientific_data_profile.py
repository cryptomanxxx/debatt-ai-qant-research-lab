"""CL-17 descriptive H1 dataset scientific profile; NO MODEL / NO INFERENCE.

Reads already archived CL-13 records only after CL-16 checksum verification.
It describes sampling bias and cross-seed identity overlap, not model accuracy.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from cl16_pinned_dataset_audit import audit_repository_archive

HERE = Path(__file__).resolve().parent
DATA = HERE / "frozen_data_candidate" / "cl13"
SEEDS = (101, 102, 103, 104, 105)
TASKS = ("T1", "T2", "T3", "T4")
TARGET_BERNOULLI = {"T1": .125, "T2": .25, "T3": .375, "T4": .5}


def describe_archived_data() -> dict:
    verified = audit_repository_archive()
    if verified["decision"] != "NO_GO":
        raise ValueError("CL-17: input receipt was promoted")
    rows = []
    across_seeds = {}
    for seed in SEEDS:
        tasks = json.loads((DATA / f"seed-{seed}.json").read_text(encoding="ascii"))
        all_seed_features = set()
        for task in tasks:
            task_id = task["task_id"]
            for partition, size in (("train", 128), ("test", 64)):
                items = task[partition]
                bit_ones = [sum(row[0][bit] for row in items) for bit in range(16)]
                xor_pairs = Counter(
                    "".join(str(bit) for bit in row[0][:2]) for row in items
                )
                identities = {tuple(row[0]) for row in items}
                all_seed_features.update(identities)
                rows.append({
                    "seed": seed, "task": task_id, "partition": partition,
                    "records": len(items),
                    "class_0": sum(row[1] == 0 for row in items),
                    "class_1": sum(row[1] == 1 for row in items),
                    "mean_active_bits_per_record": round(sum(bit_ones) / size, 6),
                    "mean_activity_fraction": round(sum(bit_ones) / (16 * size), 6),
                    "observed_bit_0_rate": round(bit_ones[0] / size, 6),
                    "observed_bit_1_rate": round(bit_ones[1] / size, 6),
                    "xor_input_pair_counts": dict(sorted(xor_pairs.items())),
                    "unconditioned_bernoulli_reference": TARGET_BERNOULLI[task_id],
                })
        across_seeds[seed] = all_seed_features
    overlap = []
    for i, first in enumerate(SEEDS):
        for second in SEEDS[i + 1:]:
            # Overlap across independently seeded runs is descriptive, NOT
            # leakage inside any one train/test partition.
            overlap.append({
                "seed_a": first, "seed_b": second,
                "common_feature_vectors": len(across_seeds[first] & across_seeds[second]),
            })
    if len(rows) != 40 or sum(row["records"] for row in rows) != 3840:
        raise ValueError("CL-17: verified data profile size drift")
    if any(row["class_0"] != row["class_1"] for row in rows):
        raise ValueError("CL-17: class balance drift")
    return {
        "decision": "NO_GO",
        "kind": "CL17_DESCRIPTIVE_H1_DATA_PROFILE",
        "total_records": 3840,
        "strata": rows,
        "between_seed_overlap": overlap,
        "sampling_caveat": "class_balanced_rejection_sampling_conditions_bernoulli_input_distribution",
        "interpretation": (
            "Within-seed train/test are disjoint by verified identity; "
            "cross-seed overlap is allowed because runs are independent. "
            "Activity shift and sample distributions are descriptive, "
            "not causal effects, learnability or generalization evidence."
        ),
        "heldout_used_for_hyperparameter_tuning": False,
        "model_imported": False,
        "training_authorized": False,
        "inference_authorized": False,
        "scientific_data_freeze_approved": False,
        "physical_energy": "NOT_MEASURED",
    }


if __name__ == "__main__":
    print(json.dumps(describe_archived_data(), sort_keys=True))
