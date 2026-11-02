"""CL-10J six-component source-event collector candidate (NOT YET WIRED).

No model imports, training, inference, data generation or energy measurement.
This collector cannot authenticate event origin and MUST NOT be serialized as
OBSERVED_SOURCE_EVENTS until reviewed model-source hook integration exists.
"""
from __future__ import annotations

from dataclasses import dataclass, field

ARMS = ("dense_sgd", "local_hybrid", "sparse_local_hybrid", "sparse_shared_index")
TASKS = ("T1", "T2", "T3", "T4")
CHECKPOINTS = ("pretrain", "post_T1", "post_T2", "post_T3", "post_T4")
EVENT_FIELDS = (
    "logical_parameter_reads", "logical_parameter_writes", "input_visits",
    "active_index_visits", "control_events", "nonlinear_events",
)
# Explicit source-hook names bind *which type of operation* can be counted.
# They are proposals, not claims that the hooks are installed.
SITES = {
    "w1_forward_read": "logical_parameter_reads",
    "w2_forward_read": "logical_parameter_reads",
    "w1_update_read": "logical_parameter_reads",
    "w2_update_read": "logical_parameter_reads",
    "w1_update_write": "logical_parameter_writes",
    "w2_update_write": "logical_parameter_writes",
    "input_scan": "input_visits",
    "active_index_read": "active_index_visits",
    "active_index_write": "active_index_visits",
    "branch_check": "control_events",
    "tanh_call": "nonlinear_events",
    "softmax_exp_call": "nonlinear_events",
    "loss_log_call": "nonlinear_events",
}


def _require(ok: bool, why: str) -> None:
    if not ok:
        raise ValueError("CL-10J NO_GO: " + why)


@dataclass
class CandidateEventCollector:
    arm: str
    seed: int
    task: str
    phase: str
    checkpoint: str
    _counts: dict[str, int] = field(
        default_factory=lambda: dict.fromkeys(EVENT_FIELDS, 0),
        init=False, repr=False,
    )

    def __post_init__(self) -> None:
        _require(type(self.arm) is str and self.arm in ARMS, "unknown arm")
        _require(type(self.seed) is int and self.seed in (101, 102, 103, 104, 105),
                 "unknown seed")
        _require(type(self.task) is str and self.task in TASKS, "unknown task")
        _require(self.phase in ("train", "heldout_inference"), "unknown phase")
        _require(self.checkpoint in CHECKPOINTS, "unknown checkpoint")
        if self.phase == "train":
            _require(self.checkpoint == "post_" + self.task,
                     "training checkpoint mismatch")

    def visit(self, site: str) -> None:
        """One proposed source event per call; no arbitrary bulk counts."""
        _require(type(site) is str and site in SITES, "unrecognized source hook")
        field_name = SITES[site]
        _require(self._counts[field_name] < 2**32 - 1, "counter overflow")
        self._counts[field_name] += 1

    def snapshot(self) -> dict:
        """Explicitly UNVERIFIED until authentic model hooks are independently audited."""
        return {
            "arm": self.arm, "seed": self.seed,
            "task": self.task, "phase": self.phase,
            "checkpoint": self.checkpoint,
            "provenance": "CANDIDATE_UNVERIFIED_SOURCE_HOOKS",
            "source_hooks_integrated": False,
            "training_authorized": False,
            "inference_authorized": False,
            "physical_energy": "NOT_MEASURED",
            **self._counts,
        }
