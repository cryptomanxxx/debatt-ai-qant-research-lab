"""CL-5C exact analytical control for the balanced-XOR sampling bias.

Pure mathematics only: no dataset generation, model execution or training.
The derivation assumes independent Bernoulli(p) feature proposals before
finite-population global-uniqueness rejection is applied.
"""
from __future__ import annotations

from fractions import Fraction

from integrations.continual_learning.cl2_data_candidate import ACTIVITY_RATES, INPUT_SIZE

LABEL_BITS = 2


def balanced_xor_proposal_moments(numerator: int, denominator: int) -> dict:
    """Exact proposal-level moments after enforcing equal XOR classes.

    For X0,X1 ~ iid Bernoulli(p), P(XOR=1) = 2p(1-p).
    Given XOR=1, the label bits have exactly one active bit.
    Given XOR=0, E[X0+X1] = 2p^2 / (p^2 + (1-p)^2).
    A 50/50 label mix alters the mean; the other bits stay iid Bernoulli(p)
    *before* global-uniqueness conditioning.
    """
    if (type(numerator) is not int or type(denominator) is not int or
            denominator <= 0 or not 0 < numerator < denominator):
        raise ValueError("require integer numerator/denominator for 0 < p < 1")
    p = Fraction(numerator, denominator)
    xor_one = 2 * p * (1 - p)
    xor_zero = 1 - xor_one
    label_bits_if_zero = 2 * p * p / xor_zero
    balanced_label_bits = (1 + label_bits_if_zero) / 2
    original_mean = INPUT_SIZE * p
    balanced_mean = (INPUT_SIZE - LABEL_BITS) * p + balanced_label_bits

    def ratio(v: Fraction) -> dict:
        return {"numerator": v.numerator, "denominator": v.denominator}

    return {
        "kind": "CL5C_ANALYTICAL_PROPOSAL_ONLY",
        "decision": "NO_GO",
        "p": ratio(p),
        "unconditioned_xor_one_probability": ratio(xor_one),
        "unconditioned_mean_active_bits": ratio(original_mean),
        "balanced_xor_mean_active_bits_before_uniqueness": ratio(balanced_mean),
        "balanced_xor_activity_rate_before_uniqueness": ratio(balanced_mean / INPUT_SIZE),
        "globally_unique_rejection_accounted_for": False,
        "dataset_generation_executed": False,
        "model_training_executed": False,
        "energy_measured": False,
    }


def planned_task_proposal_moments() -> dict:
    """Use the four known design probabilities, not observed task records."""
    expected = {"T1": (1, 8), "T2": (1, 4), "T3": (3, 8), "T4": (1, 2)}
    if list(ACTIVITY_RATES) != list(expected):
        raise ValueError("activity task order changed")
    for task_id, (num, den) in expected.items():
        if Fraction(str(ACTIVITY_RATES[task_id])) != Fraction(num, den):
            raise ValueError(f"{task_id}: activity probability changed")
    return {task: balanced_xor_proposal_moments(*ratio) for task, ratio in expected.items()}
