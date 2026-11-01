"""CL-5F analytical draw-count floor for class quotas, with no sampling.

The expected-draw bound applies to an *uncapped hypothetical* Bernoulli
proposal stream that fills both XOR-class quotas. Real generation is capped,
deduplicated, and can fail; this is not a predicted runtime of actual code.
"""
from __future__ import annotations

from fractions import Fraction

from integrations.continual_learning.cl2_data_candidate import ACTIVITY_RATES


def _ratio(value: Fraction) -> dict:
    return {"numerator": value.numerator, "denominator": value.denominator}


def uncapped_class_quota_draw_floor(
    numerator: int, denominator: int, *, train_per_task: int = 128,
    test_per_task: int = 64,
) -> dict:
    """Lower bound on expected draws until both balanced quotas are filled.

    If class c has proposal probability P_c and requires q_c hits, the
    negative-binomial time for q_c proposals of that class has mean q_c/P_c.
    Completion of *both* quotas cannot occur before either individual quota;
    hence E[T_both] >= max(q_0/P_0, q_1/P_1). This is valid even if later
    deduplication rejects proposals, but only for hypothetical uncapped
    completion, not for a finite-attempt stopped algorithm.
    """
    if (type(numerator) is not int or type(denominator) is not int or
            denominator <= 0 or not 0 < numerator < denominator):
        raise ValueError("require integer p in (0,1)")
    if (type(train_per_task) is not int or type(test_per_task) is not int or
            train_per_task <= 0 or test_per_task <= 0 or
            train_per_task % 2 or test_per_task % 2):
        raise ValueError("require positive even train/test quotas")
    p = Fraction(numerator, denominator)
    prob_one = 2 * p * (1 - p)
    prob_zero = 1 - prob_one
    rows = []
    total_floor = Fraction(0)
    for partition, count in (("train", train_per_task), ("test", test_per_task)):
        quota = count // 2
        class_zero_mean = quota / prob_zero
        class_one_mean = quota / prob_one
        floor = max(class_zero_mean, class_one_mean)
        total_floor += floor
        rows.append({
            "partition": partition,
            "requested_accepted_records": count,
            "per_class_quota": quota,
            "expected_draws_to_see_quota_of_zero": _ratio(class_zero_mean),
            "expected_draws_to_see_quota_of_one": _ratio(class_one_mean),
            "lower_bound_expected_draws_if_uncapped_completion": _ratio(floor),
        })
    return {
        "kind": "CL5F_UNCAPPED_CLASS_QUOTA_EXPECTATION_FLOOR_ONLY",
        "decision": "NO_GO",
        "nominal_p": _ratio(p),
        "xor_class_probabilities": {"0": _ratio(prob_zero), "1": _ratio(prob_one)},
        "partitions": rows,
        "lower_bound_expected_total_draws_for_task": _ratio(total_floor),
        "accepted_record_count": train_per_task + test_per_task,
        "finite_attempt_cap_modelled": False,
        "global_uniqueness_modelled": False,
        "generator_success_probability_estimated": False,
        "dataset_generation_executed": False,
        "model_training_executed": False,
        "energy_measured": False,
    }


def planned_task_draw_floors() -> dict:
    design = {"T1": (1, 8), "T2": (1, 4), "T3": (3, 8), "T4": (1, 2)}
    if list(ACTIVITY_RATES) != list(design):
        raise ValueError("CL-5F task design changed")
    if any(Fraction(str(ACTIVITY_RATES[t])) != Fraction(*numbers)
           for t, numbers in design.items()):
        raise ValueError("CL-5F nominal activity drift")
    return {task: uncapped_class_quota_draw_floor(*numbers)
            for task, numbers in design.items()}
