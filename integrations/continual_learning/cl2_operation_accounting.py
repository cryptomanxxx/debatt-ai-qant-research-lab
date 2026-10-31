"""Static, equal-scope algorithmic accounting for the exact CL-2 Python loops.

NOT hardware counters, timing, DRAM traffic, memory energy or measurement.
The original local model uses a dense forward pass; the sparse ablation does not.
"""
from __future__ import annotations

from dataclasses import dataclass

from integrations.continual_learning.cl2_mlp_baseline import (
    HIDDEN_SIZE,
    INPUT_SIZE,
    N_CLASSES,
)


@dataclass(frozen=True)
class StepProfile:
    arm: str
    input_events: int
    forward_weight_terms: int
    weight_reads: int
    weight_writes: int
    bias_reads: int
    bias_writes: int
    input_feature_checks: int

    @property
    def total_parameter_reads(self) -> int:
        return self.weight_reads + self.bias_reads

    @property
    def total_parameter_writes(self) -> int:
        return self.weight_writes + self.bias_writes


def estimate_step(arm: str, input_events: int) -> StepProfile:
    """Count Python-level logical parameter accesses for ONE update step.

    Scope for both: full forward W1/W2+biases, any downstream read of W2
    for hidden gradients, every weight/bias read-modify-write update, and
    explicit input indexing/scanning. This is a static source-level model;
    it must be updated if implementation loops change. It does not count
    hidden-state memory, cache locality, math operations or physical energy.
    """
    if type(input_events) is not int or not 0 <= input_events <= INPUT_SIZE:
        raise ValueError("input_events must be an integer in [0, 16]")
    if arm not in ("dense_sgd", "local_hybrid", "sparse_local_hybrid", "sparse_shared_index"):
        raise ValueError("unknown arm")
    first = INPUT_SIZE * HIDDEN_SIZE  # 128: dense forward in original arms
    second = HIDDEN_SIZE * N_CLASSES  # 16
    forward = first + second
    if arm == "dense_sgd":
        # Backprop reads W2 once to propagate hidden errors; SGD updates
        # every W1/W2 parameter by read-modify-write.
        weights_read = forward + second + first + second
        weights_written = first + second
        input_checks = first + first  # forward uses x; dW1 uses x again
    elif arm == "local_hybrid":
        # Local model's inherited _forward is still dense.
        # The update loops are sparse only for hidden W1 weight writes.
        weights_read = forward + second + input_events * HIDDEN_SIZE
        weights_written = second + input_events * HIDDEN_SIZE
        input_checks = first + INPUT_SIZE  # dense forward and active scan
    else:
        # Original sparse subclass scans twice; shared-index variant once.
        # In both, only W1 forward weight visits become activity-dependent.
        forward = input_events * HIDDEN_SIZE + second
        weights_read = forward + second + input_events * HIDDEN_SIZE
        weights_written = second + input_events * HIDDEN_SIZE
        input_checks = 2 * INPUT_SIZE if arm == "sparse_local_hybrid" else INPUT_SIZE
    bias_forward = HIDDEN_SIZE + N_CLASSES
    bias_updates = HIDDEN_SIZE + N_CLASSES
    return StepProfile(
        arm=arm,
        input_events=input_events,
        forward_weight_terms=forward,
        weight_reads=weights_read,
        weight_writes=weights_written,
        bias_reads=bias_forward + bias_updates,
        bias_writes=bias_updates,
        input_feature_checks=input_checks,
    )
