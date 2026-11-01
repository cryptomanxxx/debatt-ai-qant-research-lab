"""CL-5E exact class-conditioned proposal concentration; no sampling.

This computes probabilities of two *independent with-replacement proposals*
matching within a fixed XOR class. It is not the collision rate of the actual
globally deduplicated, class-balanced sequential rejection sampler.
"""
from __future__ import annotations

from fractions import Fraction
from integrations.continual_learning.cl2_data_candidate import ACTIVITY_RATES
from integrations.continual_learning.cl5d_xor_support_audit import xor_support_by_weight


def _ratio(value: Fraction) -> dict:
    return {"numerator": value.numerator, "denominator": value.denominator}


def conditional_collision_proxy(numerator: int, denominator: int) -> dict:
    """Exact Bernoulli proposal mass and independent class-conditional collision.

    Each vector of Hamming weight k has mass p**k * (1-p)**(16-k).
    Within XOR class c: collision_c = sum_x P(x|c)**2.

    Two independently proposed vectors must be sampled from the *same* class;
    this is a distribution-concentration diagnostic only.
    """
    if (type(numerator) is not int or type(denominator) is not int or
            denominator <= 0 or not 0 < numerator < denominator):
        raise ValueError("require integers with 0 < numerator < denominator")
    p = Fraction(numerator, denominator)
    audit = xor_support_by_weight()
    n = audit["feature_count"]
    class_mass = [Fraction(0), Fraction(0)]
    squared_mass = [Fraction(0), Fraction(0)]
    for row in audit["weights"]:
        k = row["active_bits"]
        vector_mass = p ** k * (1 - p) ** (n - k)
        for label in (0, 1):
            multiplicity = row[f"label_{'zero' if label == 0 else 'one'}_support"]
            class_mass[label] += multiplicity * vector_mass
            squared_mass[label] += multiplicity * vector_mass * vector_mass
    if sum(class_mass) != 1 or min(class_mass) <= 0:
        raise ValueError("invalid class-conditioned mass")
    collision = [squared_mass[c] / class_mass[c] ** 2 for c in (0, 1)]
    uniform = Fraction(1, audit["support_per_class"]["0"])
    if any(value < uniform or value > 1 for value in collision):
        raise ValueError("invalid collision concentration")
    return {
        "kind": "CL5E_INDEPENDENT_PROPOSAL_CONCENTRATION_ONLY",
        "decision": "NO_GO",
        "proposal_p": _ratio(p),
        "class_probabilities": {"0": _ratio(class_mass[0]), "1": _ratio(class_mass[1])},
        "same_class_independent_collision_probabilities": {
            "0": _ratio(collision[0]), "1": _ratio(collision[1])
        },
        "uniform_support_collision_reference": _ratio(uniform),
        "global_uniqueness_or_rejection_attempts_modelled": False,
        "cross_task_collision_risk_modelled": False,
        "dataset_generation_executed": False,
        "model_training_executed": False,
        "energy_measured": False,
    }


def planned_collision_proxies() -> dict:
    """Nominal CL task settings; calculate exact fractions without samples."""
    proposed = {"T1": (1, 8), "T2": (1, 4), "T3": (3, 8), "T4": (1, 2)}
    if list(ACTIVITY_RATES) != list(proposed):
        raise ValueError("task order changed")
    if any(Fraction(str(ACTIVITY_RATES[task])) != Fraction(*numbers)
           for task, numbers in proposed.items()):
        raise ValueError("nominal activity rates changed")
    return {task: conditional_collision_proxy(*numbers) for task, numbers in proposed.items()}
