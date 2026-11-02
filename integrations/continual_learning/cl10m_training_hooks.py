"""CL-10M opt-in training-phase source hook helpers (CANDIDATE / NO_GO).

Never constructs models, accesses datasets, trains, predicts or measures energy.
The caller's original noninstrumented path must remain unchanged.
"""
from __future__ import annotations

from integrations.continual_learning.cl10l_forward_hooks import attached_candidate


def observed_local_step_active(model: object, x: tuple[int, ...]) -> list[int]:
    """Instrument the actual local-step active-list build once, not a second scan."""
    if not attached_candidate(model):
        raise ValueError("CL-10M NO_GO: attached candidate observer required")
    active = []
    for i, bit in enumerate(x):
        model._cl10k_source_observer.visit("input_scan")
        model._cl10k_source_observer.visit("branch_check")
        if bit:
            active.append(i)
            model._cl10k_source_observer.visit("active_index_write")
    return active


def record_nonlinear_value(model: object, site: str, value: float) -> float:
    """Record one completed exp/log call without recalculating the value."""
    if site not in ("softmax_exp_call", "loss_log_call"):
        raise ValueError("CL-10M NO_GO: unsupported nonlinear site")
    if not attached_candidate(model):
        raise ValueError("CL-10M NO_GO: attached candidate observer required")
    model._cl10k_source_observer.visit(site)
    return value
