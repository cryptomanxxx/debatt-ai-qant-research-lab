"""CL-3E non-executable measurement ledger schema and logical-cost projections.

All projected counts are source-level estimates, never observed CPU/DRAM
traffic, energy, timing, or physical measurements. No dataset or model run.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

from integrations.continual_learning.cl2_operation_accounting import estimate_step
from integrations.continual_learning.cl2_payload_proxy import (
    ProxyAssumptions,
    estimate_payload_proxy,
)

SCOPE_VERSION = "cl3e-source-loop-estimate-v0"
NOT_MEASURED = "NOT_MEASURED"


@dataclass(frozen=True)
class SourceEstimate:
    scope_version: str
    phase: str
    arm: str
    step_count_assumed: int
    active_inputs_per_step_assumed: int
    parameter_scalar_bytes_assumed: int
    input_scalar_bytes_assumed: int
    active_index_bytes_assumed: int
    forward_weight_terms: int
    logical_parameter_reads: int
    logical_parameter_writes: int
    input_checks: int
    logical_payload_bytes: int
    physical_energy_joules: str = NOT_MEASURED
    cpu_wall_seconds: str = NOT_MEASURED
    dram_bytes: str = NOT_MEASURED


def project_training_only(
    arm: str, *,
    step_count: int,
    active_inputs_per_step: int,
    assumptions: ProxyAssumptions = ProxyAssumptions(),
) -> SourceEstimate:
    """Only hypothetical equal-activity repeated steps; NO execution.

    Training projections count single-step loops in CL-2 exactly as presently
    modeled. Actual activity varies by example and requires a future reviewed
    runner to record counts. This cannot estimate heldout inference.
    """
    if type(step_count) is not int or not 0 <= step_count <= 512:
        raise ValueError("proposed single-seed/arm ceiling is 512 updates")
    if not isinstance(assumptions, ProxyAssumptions):
        raise ValueError("invalid payload assumptions")
    profile = estimate_step(arm, active_inputs_per_step)
    payload = estimate_payload_proxy(arm, active_inputs_per_step, assumptions)
    return SourceEstimate(
        scope_version=SCOPE_VERSION,
        phase="HYPOTHETICAL_TRAINING_STEP_PROJECTION",
        arm=arm,
        step_count_assumed=step_count,
        active_inputs_per_step_assumed=active_inputs_per_step,
        parameter_scalar_bytes_assumed=assumptions.parameter_scalar_bytes,
        input_scalar_bytes_assumed=assumptions.input_scalar_bytes,
        active_index_bytes_assumed=assumptions.active_index_bytes,
        forward_weight_terms=step_count * profile.forward_weight_terms,
        logical_parameter_reads=step_count * profile.total_parameter_reads,
        logical_parameter_writes=step_count * profile.total_parameter_writes,
        input_checks=step_count * profile.input_feature_checks,
        logical_payload_bytes=step_count * payload.total_logical_payload_bytes,
    )


def measurement_scope_manifest() -> dict:
    """Required separation of source-level projections and unmeasured terms."""
    return {
        "scope_version": SCOPE_VERSION,
        "training_accounting": "static_source_loop_projection_only",
        "heldout_inference_accounting": "static_predict_source_loop_projection_only_cl3n",
        "index_scans": "input_checks_from_cl2_operation_accounting",
        "active_index_visits": "payload_proxy_assumption_only",
        "bias_accesses": "included_in_logical_parameter_counts",
        "nonlinearities": "NOT_COUNTED",
        "hidden_state_accesses": "NOT_COUNTED",
        "cache_and_dram_traffic": NOT_MEASURED,
        "data_conversion_and_optical_io": NOT_MEASURED,
        "cpu_wall_seconds": NOT_MEASURED,
        "physical_energy_joules": NOT_MEASURED,
        "trained_models": False,
        "experiment_authorized": False,
    }


def projection_dict(estimate: SourceEstimate) -> dict:
    return asdict(estimate)
