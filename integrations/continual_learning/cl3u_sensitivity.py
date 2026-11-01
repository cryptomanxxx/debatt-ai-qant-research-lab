"""CL-3U algebraic architecture/width sensitivity; NO model execution.

The 16x8x2 reference reproduces existing CL-2/CL-3 source-loop proxies.
Other dimensions are *counterfactual loop extrapolations*, not implementations.
"""
from __future__ import annotations

from dataclasses import dataclass

from integrations.continual_learning.cl2_payload_proxy import ProxyAssumptions
from integrations.continual_learning.cl3e_measurement_ledger import project_training_only
from integrations.continual_learning.cl3n_inference_accounting import estimate_prediction

ARMS = ("dense_sgd", "local_hybrid", "sparse_local_hybrid", "sparse_shared_index")
PHASES = ("training", "inference")


@dataclass(frozen=True)
class Shape:
    inputs: int = 16
    hidden: int = 8
    outputs: int = 2

    def __post_init__(self):
        if any(type(n) is not int or not 1 <= n <= 256 for n in (self.inputs, self.hidden, self.outputs)):
            raise ValueError("shape dimensions must be positive integers <= 256")


def project(arm: str, phase: str, shape: Shape, active: int, widths: ProxyAssumptions) -> int:
    """Assumed packed logical access bytes, not physical traffic or joules."""
    if arm not in ARMS or phase not in PHASES:
        raise ValueError("unknown arm or phase")
    if not isinstance(shape, Shape) or not isinstance(widths, ProxyAssumptions):
        raise ValueError("invalid assumptions")
    if type(active) is not int or not 0 <= active <= shape.inputs:
        raise ValueError("activity exceeds shape")
    n, h, c = shape.inputs, shape.hidden, shape.outputs
    first, second = n * h, h * c
    sparse = arm in ("sparse_local_hybrid", "sparse_shared_index")
    if phase == "training":
        if arm == "dense_sgd":
            param = 3 * first + 4 * second + 3 * (h + c)
            input_visits, index_visits = 2 * first, 0
        elif arm == "local_hybrid":
            param = first + 3 * second + 2 * active * h + 3 * (h + c)
            input_visits, index_visits = first + n, 9 * active
        else:
            param = 3 * active * h + 3 * second + 3 * (h + c)
            input_visits = 2 * n if arm == "sparse_local_hybrid" else n
            index_visits = (18 if arm == "sparse_local_hybrid" else 17) * active
    else:
        param = (active * h if sparse else first) + second + h + c
        input_visits = 2 * n if sparse else n + first
        index_visits = active * (1 + h) if sparse else 0
    return (param * widths.parameter_scalar_bytes
            + input_visits * widths.input_scalar_bytes
            + index_visits * widths.active_index_bytes)


def sensitivity() -> dict:
    """Small, preregistered deterministic scenario grid; no datasets/loops."""
    scenarios = []
    shape_grid = (Shape(8, 4, 2), Shape(), Shape(32, 16, 2))
    width_grid = (
        ProxyAssumptions(8, 1, 2),
        ProxyAssumptions(4, 1, 2),
        ProxyAssumptions(2, 1, 4),
    )
    for shape in shape_grid:
        for widths in width_grid:
            rows = []
            for phase in PHASES:
                costs = {arm: [project(arm, phase, shape, active, widths)
                              for active in range(shape.inputs + 1)] for arm in ARMS}
                dense = costs["dense_sgd"]
                for arm in ARMS[1:]:
                    flags = [cost < base for cost, base in zip(costs[arm], dense)]
                    # Do not assign a single crossover to a non-monotone curve.
                    if any(not flags[i] and flags[i + 1] for i in range(len(flags) - 1)):
                        raise ValueError("nonmonotone advantage in static scenario")
                    rows.append({
                        "phase": phase, "arm": arm,
                        "last_strict_advantage": max((i for i, flag in enumerate(flags) if flag), default=None),
                        "first_not_better": next((i for i, flag in enumerate(flags) if not flag), None),
                        "full_activity_candidate_bytes": costs[arm][-1],
                        "full_activity_dense_bytes": dense[-1],
                    })
            scenarios.append({
                "shape": {"inputs": shape.inputs, "hidden": shape.hidden, "outputs": shape.outputs},
                "widths": {"parameter": widths.parameter_scalar_bytes,
                           "input": widths.input_scalar_bytes, "index": widths.active_index_bytes},
                "comparisons": rows,
            })
    return {
        "kind": "CL3U_COUNTERFACTUAL_LOGICAL_BYTES_NOT_ENERGY",
        "scenarios": scenarios,
        "actual_training": False,
        "actual_inference": False,
        "hardware_energy_measured": False,
        "accuracy_evaluated": False,
        "caveat": "Only 16x8x2 matches audited source loops; other shapes extrapolate loop structure without implemented or trained networks.",
    }


def check_reference() -> None:
    """Cross-check exact baseline shape against independent CL-3 estimators."""
    shape = Shape()
    for widths in (ProxyAssumptions(), ProxyAssumptions(2, 1, 4)):
        for active in range(17):
            for arm in ARMS:
                train = project_training_only(arm, step_count=1, active_inputs_per_step=active, assumptions=widths)
                infer = estimate_prediction(arm, active, assumptions=widths)
                if project(arm, "training", shape, active, widths) != train.logical_payload_bytes:
                    raise AssertionError("training proxy drift")
                if project(arm, "inference", shape, active, widths) != infer.logical_payload_bytes:
                    raise AssertionError("inference proxy drift")
