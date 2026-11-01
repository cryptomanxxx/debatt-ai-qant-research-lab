"""CL-3V: lossless decomposition of assumed packed logical payloads only.

No model or dataset is instantiated. Does not infer DRAM traffic or energy.
"""
from __future__ import annotations

from integrations.continual_learning.cl2_payload_proxy import ProxyAssumptions, estimate_payload_proxy
from integrations.continual_learning.cl3e_measurement_ledger import project_training_only
from integrations.continual_learning.cl3n_inference_accounting import estimate_prediction

ARMS = ("dense_sgd", "local_hybrid", "sparse_local_hybrid", "sparse_shared_index")
PHASES = ("training", "inference")


def decompose(arm: str, phase: str, active: int, widths: ProxyAssumptions = ProxyAssumptions()) -> dict:
    """Partition CL-3E/N totals into parameter, input and index bytes."""
    if arm not in ARMS or phase not in PHASES:
        raise ValueError("unknown arm/phase")
    if not isinstance(widths, ProxyAssumptions):
        raise ValueError("invalid widths")
    if type(active) is not int or not 0 <= active <= 16:
        raise ValueError("activity outside 0..16")
    if phase == "training":
        p = estimate_payload_proxy(arm, active, widths)
        components = {
            "parameter_bytes": p.parameter_access_payload_bytes,
            "input_bytes": p.input_check_payload_bytes,
            "index_bytes": p.active_index_access_payload_bytes,
        }
        expected = project_training_only(
            arm, step_count=1, active_inputs_per_step=active, assumptions=widths
        ).logical_payload_bytes
    else:
        p = estimate_prediction(arm, active, assumptions=widths)
        components = {
            "parameter_bytes": p.parameter_reads * widths.parameter_scalar_bytes,
            "input_bytes": p.input_feature_visits * widths.input_scalar_bytes,
            "index_bytes": (p.active_index_writes + p.active_index_reads) * widths.active_index_bytes,
        }
        expected = p.logical_payload_bytes
    total = sum(components.values())
    if total != expected:
        raise AssertionError("decomposition does not reconcile with baseline")
    return {"arm": arm, "phase": phase, "active": active,
            **components, "total_bytes": total}


def component_report() -> dict:
    """Complete fixed-shape activity series, phase-separated, same assumptions."""
    widths = ProxyAssumptions()
    rows = [
        decompose(arm, phase, active, widths)
        for phase in PHASES for active in range(17) for arm in ARMS
    ]
    return {
        "kind": "CL3V_ASSUMED_LOGICAL_PAYLOAD_COMPONENTS_NOT_ENERGY",
        "shape": "16x8x2",
        "assumed_byte_widths": {
            "parameter": widths.parameter_scalar_bytes,
            "input": widths.input_scalar_bytes,
            "index": widths.active_index_bytes,
        },
        "rows": rows,
        "training_executed": False,
        "inference_executed": False,
        "energy_measured": False,
        "note": "Components reconcile source-level logical byte counts, not real memory traffic, time or joules.",
    }
