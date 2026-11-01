"""CL-5D exact combinatorial support audit; no samples or model execution.

Counts possible 16-bit vectors by Hamming weight and XOR label. Support
counts do not establish that rejection sampling is efficient or unbiased.
"""
from __future__ import annotations

from math import comb

from integrations.continual_learning.cl2_data_candidate import ACTIVITY_RATES, INPUT_SIZE
from integrations.continual_learning.cl2_shared_rule_candidate import GLOBAL_LABEL_PAIR

LABEL_BIT_COUNT = 2


def _choose(n: int, k: int) -> int:
    return comb(n, k) if 0 <= k <= n else 0


def xor_support_by_weight() -> dict:
    """Number of *possible* feature vectors at each activity weight and label.

    Label 1 has exactly one active XOR bit; label 0 has zero or two.
    The remaining INPUT_SIZE-2 bits are unrestricted.
    """
    if INPUT_SIZE != 16 or GLOBAL_LABEL_PAIR != (0, 1):
        raise ValueError("CL-5D support formula needs the reviewed 16-bit XOR pair")
    remainder = INPUT_SIZE - LABEL_BIT_COUNT
    rows = []
    class_zero = class_one = 0
    for active_bits in range(INPUT_SIZE + 1):
        count_one = 2 * _choose(remainder, active_bits - 1)
        count_zero = (_choose(remainder, active_bits) +
                      _choose(remainder, active_bits - 2))
        rows.append({
            "active_bits": active_bits,
            "label_zero_support": count_zero,
            "label_one_support": count_one,
            "total_support": count_zero + count_one,
        })
        class_zero += count_zero
        class_one += count_one
    expected = 2 ** (INPUT_SIZE - 1)
    if class_zero != expected or class_one != expected:
        raise ValueError("XOR support conservation failed")
    if sum(row["total_support"] for row in rows) != 2 ** INPUT_SIZE:
        raise ValueError("total feature-space conservation failed")
    return {
        "kind": "CL5D_EXACT_FEATURE_SUPPORT_ONLY",
        "decision": "NO_GO",
        "feature_count": INPUT_SIZE,
        "label_pair": list(GLOBAL_LABEL_PAIR),
        "support_per_class": {"0": class_zero, "1": class_one},
        "weights": rows,
        "observed_samples": 0,
        "dataset_generation_executed": False,
        "model_training_executed": False,
        "energy_measured": False,
    }


def planned_quota_capacity(*, train_per_task: int = 128,
                           test_per_task: int = 64) -> dict:
    """Check a *necessary* global support bound, not sampling feasibility.

    Each of the four tasks requests balanced classes across both partitions.
    Class support is shared across tasks because generator input uniqueness is
    global within a seed; per-task duplication of capacity is not allowed.
    """
    if (type(train_per_task) is not int or type(test_per_task) is not int or
            train_per_task <= 0 or test_per_task <= 0 or
            train_per_task % 2 or test_per_task % 2):
        raise ValueError("train/test quotas must be positive even integers")
    if list(ACTIVITY_RATES) != ["T1", "T2", "T3", "T4"]:
        raise ValueError("task ordering changed")
    capacity = xor_support_by_weight()["support_per_class"]["0"]
    required = len(ACTIVITY_RATES) * (train_per_task + test_per_task) // 2
    if required > capacity:
        raise ValueError("global unique class quota exceeds feature support")
    return {
        "kind": "CL5D_NECESSARY_CAPACITY_ONLY",
        "decision": "NO_GO",
        "required_distinct_per_class_across_all_tasks": required,
        "available_distinct_per_class": capacity,
        "necessary_capacity_check_passed": True,
        "sampling_feasibility_proven": False,
        "conditional_distribution_audited": False,
        "dataset_generation_executed": False,
        "model_training_executed": False,
        "energy_measured": False,
    }
