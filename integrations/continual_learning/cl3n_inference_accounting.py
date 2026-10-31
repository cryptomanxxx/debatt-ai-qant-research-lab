"""CL-3N static prediction-path accounting; no model instantiated or called.

Counts Python source-level accesses for predict(), not energy, time,
hardware memory traffic, Python object allocations or measured inference.
"""
from __future__ import annotations

from dataclasses import dataclass

from integrations.continual_learning.cl2_mlp_baseline import (
    HIDDEN_SIZE, INPUT_SIZE, N_CLASSES,
)
from integrations.continual_learning.cl2_payload_proxy import ProxyAssumptions

INFERENCE_SCOPE_VERSION = "cl3n-predict-source-loop-v0"
ARMS = ("dense_sgd", "local_hybrid", "sparse_local_hybrid", "sparse_shared_index")


@dataclass(frozen=True)
class InferenceEstimate:
    scope_version: str
    arm: str
    active_input_bits_assumed: int
    forward_weight_terms: int
    parameter_reads: int
    parameter_writes: int
    input_feature_visits: int
    active_index_writes: int
    active_index_reads: int
    tanh_evaluations: int
    output_score_comparisons_assumed: int
    logical_payload_bytes: int
    parameter_scalar_bytes_assumed: int
    input_scalar_bytes_assumed: int
    active_index_bytes_assumed: int
    physical_energy_joules: str = "NOT_MEASURED"
    dram_bytes: str = "NOT_MEASURED"


def estimate_prediction(
    arm: str,
    active_input_bits: int,
    assumptions: ProxyAssumptions = ProxyAssumptions(),
) -> InferenceEstimate:
    """Static model for standalone predict() (not step/update).

    Dense prediction validates 16 inputs, then W1 forward visits 128 inputs.
    Sparse predict() validates and scans inputs separately, and reuses each
    active index across eight hidden neurons. Shared-index optimization is
    step-only: standalone predict() still uses inherited sparse _forward().
    """
    if arm not in ARMS:
        raise ValueError("unknown prediction arm")
    if type(active_input_bits) is not int or not 0 <= active_input_bits <= INPUT_SIZE:
        raise ValueError("active bits outside [0,16]")
    if not isinstance(assumptions, ProxyAssumptions):
        raise ValueError("invalid payload widths")
    sparse = arm in ("sparse_local_hybrid", "sparse_shared_index")
    first = active_input_bits * HIDDEN_SIZE if sparse else INPUT_SIZE * HIDDEN_SIZE
    second = HIDDEN_SIZE * N_CLASSES
    weight_terms = first + second
    parameter_reads = weight_terms + HIDDEN_SIZE + N_CLASSES
    input_visits = (2 * INPUT_SIZE) if sparse else INPUT_SIZE + INPUT_SIZE * HIDDEN_SIZE
    index_writes = active_input_bits if sparse else 0
    index_reads = active_input_bits * HIDDEN_SIZE if sparse else 0
    bytes_total = (
        parameter_reads * assumptions.parameter_scalar_bytes
        + input_visits * assumptions.input_scalar_bytes
        + (index_writes + index_reads) * assumptions.active_index_bytes
    )
    return InferenceEstimate(
        scope_version=INFERENCE_SCOPE_VERSION,
        arm=arm,
        active_input_bits_assumed=active_input_bits,
        forward_weight_terms=weight_terms,
        parameter_reads=parameter_reads,
        parameter_writes=0,
        input_feature_visits=input_visits,
        active_index_writes=index_writes,
        active_index_reads=index_reads,
        tanh_evaluations=HIDDEN_SIZE,
        output_score_comparisons_assumed=N_CLASSES - 1,
        logical_payload_bytes=bytes_total,
        parameter_scalar_bytes_assumed=assumptions.parameter_scalar_bytes,
        input_scalar_bytes_assumed=assumptions.input_scalar_bytes,
        active_index_bytes_assumed=assumptions.active_index_bytes,
    )
