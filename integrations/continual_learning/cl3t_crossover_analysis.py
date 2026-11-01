"""CL-3T exact discrete source-proxy crossover analysis; NO experiment.

Compares hypothetical logical access payload bytes to dense source-loop
estimates for each integer activity level 0..16. Never estimates joules.
"""
from __future__ import annotations

ARMS = ("local_hybrid", "sparse_local_hybrid", "sparse_shared_index")
PHASE_METRICS = {
    "training": "training_logical_payload_bytes_per_assumed_update",
    "inference": "inference_logical_payload_bytes_per_assumed_prediction",
}


def analyze_crossover(preview: dict) -> dict:
    """Find first point at/above dense after a strictly cheaper region.

    Integer activity levels are the complete 16-bit candidate input domain.
    An equality is not a savings. A non-monotone comparison is refused rather
    than summarized with an incorrectly simple single crossover.
    """
    rows = preview.get("series")
    if not isinstance(rows, list) or [x.get("active_input_bits") for x in rows] != list(range(17)):
        raise ValueError("expected complete ordered activity sweep 0..16")
    results = []
    for phase, key in PHASE_METRICS.items():
        for arm in ARMS:
            savings = []
            for point in rows:
                by_arm = point["by_arm"]
                dense = by_arm["dense_sgd"][key]
                candidate = by_arm[arm][key]
                if type(dense) is not int or type(candidate) is not int or dense <= 0 or candidate < 0:
                    raise ValueError("invalid static payload")
                savings.append(candidate < dense)
            if any(savings[i] is False and savings[i + 1] is True for i in range(16)):
                raise ValueError("non-monotone cost advantage: cannot report single crossover")
            beneficial = [i for i, value in enumerate(savings) if value]
            last = max(beneficial) if beneficial else None
            first_not_better = next((i for i, flag in enumerate(savings) if not flag), None)
            results.append({
                "phase": phase,
                "arm": arm,
                "beneficial_activity_levels": beneficial,
                "last_strictly_better_activity": last,
                "first_not_better_activity": first_not_better,
                "at_full_activity_better_than_dense": savings[-1],
                "physical_energy_inference_allowed": False,
            })
    return {
        "kind": "CL3T_LOGICAL_PAYLOAD_CROSSOVER_NOT_ENERGY",
        "baseline": "dense_sgd",
        "input_activity_levels": list(range(17)),
        "comparisons": results,
        "no_training_or_inference_performed": True,
        "accuracy_tradeoff_verified": False,
        "note": "Breakpoints apply only to assumed packed logical accesses in this Python source model. They are not CPU/GPU/photonic energy or performance thresholds.",
    }
