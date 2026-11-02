"""CL-10K optional weight-update source hooks (PARTIAL / NO_GO).

This adapter does not instantiate or execute models. Hooks only fire when a
future separately authorized caller explicitly attaches a matching collector.
The collector remains UNVERIFIED and cannot be exported as observed H1 data.
"""
from __future__ import annotations

from integrations.continual_learning.cl10j_source_event_collector import CandidateEventCollector

CLASS_ARMS = {
    "TinyDenseMLP": "dense_sgd",
    "LocalHebbianMLP": "local_hybrid",
    "SparseForwardLocalMLP": "sparse_local_hybrid",
    "SharedIndexSparseForwardLocalMLP": "sparse_shared_index",
}
SITES = {
    "w1_update_read", "w1_update_write",
    "w2_update_read", "w2_update_write",
}


def attach_candidate_weight_observer(model: object, observer: object) -> None:
    """Attach only to the exact four planned classes; no model call occurs."""
    if type(observer) is not CandidateEventCollector:
        raise ValueError("CL-10K NO_GO: invalid observer")
    arm = CLASS_ARMS.get(type(model).__name__)
    if arm is None or arm != observer.arm:
        raise ValueError("CL-10K NO_GO: unexpected model/arm pairing")
    if getattr(model, "_cl10k_source_observer", None) is not None:
        raise ValueError("CL-10K NO_GO: duplicate observer attachment")
    setattr(model, "_cl10k_source_observer", observer)


def note_weight_update(model: object, site: str) -> None:
    if site not in SITES:
        raise ValueError("CL-10K NO_GO: unexpected source hook")
    observer = getattr(model, "_cl10k_source_observer", None)
    if observer is not None:
        if type(observer) is not CandidateEventCollector:
            raise ValueError("CL-10K NO_GO: corrupted observer")
        observer.visit(site)
