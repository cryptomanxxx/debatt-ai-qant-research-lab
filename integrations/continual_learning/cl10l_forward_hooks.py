"""CL-10L opt-in instrumented forward path. CANDIDATE, NOT VERIFIED.

Only entered with an explicitly attached CL-10J collector. Original model
forward paths remain unchanged when no collector is attached. Numerical
equivalence has NOT yet been measured or independently signed off.
"""
from __future__ import annotations

import math

from integrations.continual_learning.cl10j_source_event_collector import CandidateEventCollector


def attached_candidate(model: object) -> bool:
    observer = getattr(model, "_cl10k_source_observer", None)
    if observer is None:
        return False
    if type(observer) is not CandidateEventCollector:
        raise ValueError("CL-10L NO_GO: corrupted candidate observer")
    return True


def _site(model: object, site: str) -> None:
    model._cl10k_source_observer.visit(site)


def candidate_forward(model: object, x: tuple[int, ...], *,
                      mode: str, reused_active: list[int] | None = None
                      ) -> tuple[list[float], list[float]]:
    """Source-attached hook calls around arithmetic in a separate opt-in path.

    Constructing lists of terms preserves the same term order and sum placement
    as the existing generator expressions. This is a design claim awaiting
    independent parity testing, not established numerical equivalence.
    """
    if not attached_candidate(model):
        raise ValueError("CL-10L NO_GO: opt-in observer required")
    if mode not in ("dense", "sparse", "shared"):
        raise ValueError("CL-10L NO_GO: invalid forward mode")
    if type(x) is not tuple or len(x) != 16 or any(
        type(bit) is not int or bit not in (0, 1) for bit in x
    ):
        raise ValueError("CL-10L NO_GO: invalid input shape")
    if mode == "shared":
        if type(reused_active) is not list or reused_active != [
            i for i, bit in enumerate(x) if bit
        ]:
            raise ValueError("CL-10L NO_GO: shared active list invalid")
        active = reused_active
    elif mode == "sparse":
        active = []
        for i, bit in enumerate(x):
            _site(model, "input_scan")
            _site(model, "branch_check")
            if bit:
                active.append(i)
                _site(model, "active_index_write")
    else:
        active = None

    hidden = []
    for j in range(8):
        first_terms = []
        if mode == "dense":
            for i in range(16):
                _site(model, "input_scan")
                _site(model, "w1_forward_read")
                first_terms.append(model.w1[j][i] * x[i])
        else:
            for i in active:
                _site(model, "active_index_read")
                _site(model, "w1_forward_read")
                first_terms.append(model.w1[j][i])
        _site(model, "tanh_call")
        hidden.append(math.tanh(model.b1[j] + sum(first_terms)))
    logits = []
    for k in range(2):
        second_terms = []
        for j in range(8):
            _site(model, "w2_forward_read")
            second_terms.append(model.w2[k][j] * hidden[j])
        logits.append(model.b2[k] + sum(second_terms))
    return hidden, logits
