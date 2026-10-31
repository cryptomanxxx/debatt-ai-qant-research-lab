"""Hypothetical packed-data access-volume proxy, NOT physical memory traffic.

Maps source-level logical accesses to byte-equivalent payloads with explicit
scalar/index widths. It models no cache, reuse, allocation, timing or energy.
"""
from __future__ import annotations

from dataclasses import dataclass

from integrations.continual_learning.cl2_operation_accounting import estimate_step


@dataclass(frozen=True)
class ProxyAssumptions:
    parameter_scalar_bytes: int = 8
    input_scalar_bytes: int = 1
    active_index_bytes: int = 2

    def __post_init__(self):
        for name, value in (
            ("parameter_scalar_bytes", self.parameter_scalar_bytes),
            ("input_scalar_bytes", self.input_scalar_bytes),
            ("active_index_bytes", self.active_index_bytes),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class PayloadProxy:
    arm: str
    input_events: int
    parameter_access_payload_bytes: int
    input_check_payload_bytes: int
    active_index_access_payload_bytes: int

    @property
    def total_logical_payload_bytes(self) -> int:
        return (
            self.parameter_access_payload_bytes
            + self.input_check_payload_bytes
            + self.active_index_access_payload_bytes
        )


def estimate_payload_proxy(
    arm: str, input_events: int, assumptions: ProxyAssumptions = ProxyAssumptions()
) -> PayloadProxy:
    """Estimate *logical* access payload bytes for a single step.

    Source-level active-index list visits:
      dense_sgd: none
      local_hybrid: 1 write per active index, 8 reads per active index
      sparse_local_hybrid: 2 writes and 16 reads per active index
    Reads/writes are hypothetical compact scalars; actual Python memory
    behavior and physical bus/DRAM traffic may differ drastically.
    """
    if not isinstance(assumptions, ProxyAssumptions):
        raise ValueError("assumptions must be ProxyAssumptions")
    profile = estimate_step(arm, input_events)
    index_visits_per_event = {
        "dense_sgd": 0,
        "local_hybrid": 1 + 8,
        "sparse_local_hybrid": 2 + 16,
    }[arm]
    return PayloadProxy(
        arm=arm,
        input_events=input_events,
        parameter_access_payload_bytes=(
            profile.total_parameter_reads + profile.total_parameter_writes
        ) * assumptions.parameter_scalar_bytes,
        input_check_payload_bytes=profile.input_feature_checks * assumptions.input_scalar_bytes,
        active_index_access_payload_bytes=(
            index_visits_per_event * input_events * assumptions.active_index_bytes
        ),
    )


def sensitivity_table(input_events: int) -> tuple[PayloadProxy, ...]:
    """Deterministic range over *assumed* 2/4/8-byte packed parameters."""
    return tuple(
        estimate_payload_proxy(arm, input_events, ProxyAssumptions(parameter_scalar_bytes=width))
        for width in (2, 4, 8)
        for arm in ("dense_sgd", "local_hybrid", "sparse_local_hybrid")
    )
