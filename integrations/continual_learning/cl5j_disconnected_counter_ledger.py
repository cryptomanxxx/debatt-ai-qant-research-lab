"""CL-5J disconnected event counter ledger; NEVER runs a dataset generator.

The caller supplies abstract decision events (no feature vectors). It is not
wired to the real CL dataset generator or any CI training/data job.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ProposalCounterLedger:
    attempt_cap: int = 1_000_000
    candidate_attempts: int = field(default=0, init=False)
    class_full_rejections: int = field(default=0, init=False)
    duplicate_rejections_after_class_check: int = field(default=0, init=False)
    accepted_records: int = field(default=0, init=False)
    _pending: bool = field(default=False, init=False, repr=False)
    _finished: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        if type(self.attempt_cap) is not int or self.attempt_cap <= 0:
            raise ValueError("attempt cap must be a positive integer")

    def begin_proposal(self) -> None:
        """Mark one post-early-break proposal; does not draw any random bits."""
        if self._finished or self._pending or self.candidate_attempts >= self.attempt_cap:
            raise ValueError("proposal counter lifecycle violation")
        self.candidate_attempts += 1
        self._pending = True

    def resolve_proposal(self, *, class_full: bool,
                         duplicate_if_class_open: bool | None) -> str:
        """Resolve a proposal using generator's class-first short-circuit priority.

        When class_full is true, duplicate_if_class_open MUST be None: in the
        original generator the duplicate predicate is not evaluated.
        """
        if self._finished or not self._pending:
            raise ValueError("no pending proposal")
        if type(class_full) is not bool:
            raise ValueError("class_full must be bool")
        if class_full:
            if duplicate_if_class_open is not None:
                raise ValueError("duplicate predicate must remain unevaluated")
            self.class_full_rejections += 1
            decision = "class_full"
        else:
            if type(duplicate_if_class_open) is not bool:
                raise ValueError("duplicate_if_class_open must be bool")
            if duplicate_if_class_open:
                self.duplicate_rejections_after_class_check += 1
                decision = "duplicate"
            else:
                self.accepted_records += 1
                decision = "accepted"
        self._pending = False
        return decision

    def finish_partition(self, *, quotas_filled: bool) -> dict:
        """Close an abstract partition and report whether its cap was exhausted."""
        if type(quotas_filled) is not bool or self._pending or self._finished:
            raise ValueError("invalid partition completion")
        if not quotas_filled and self.candidate_attempts != self.attempt_cap:
            raise ValueError("incomplete partition must exhaust its attempt cap")
        if self.candidate_attempts != (
            self.accepted_records + self.class_full_rejections
            + self.duplicate_rejections_after_class_check
        ):
            raise ValueError("proposal outcome conservation violated")
        self._finished = True
        return {
            "kind": "CL5J_DISCONNECTED_COUNTER_LEDGER_ONLY",
            "candidate_attempts": self.candidate_attempts,
            "class_full_rejections": self.class_full_rejections,
            "duplicate_rejections_after_class_check": self.duplicate_rejections_after_class_check,
            "accepted_records": self.accepted_records,
            "attempt_cap_exhausted": not quotas_filled,
            "generator_connected": False,
            "dataset_generation_executed": False,
            "model_training_executed": False,
            "scientific_decision": "NO_GO",
        }
