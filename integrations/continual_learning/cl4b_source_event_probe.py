"""CL-4B isolated source-loop event probe (no models, training, or inference).

Runs instrumented loop traversals mirroring the first/second layer iteration
patterns of CL-2. This is a *probe*, not instrumentation attached to a trained
model and not CPU instruction counts, memory bus traffic or energy.
"""
from __future__ import annotations

from integrations.continual_learning.cl2_mlp_baseline import INPUT_SIZE, HIDDEN_SIZE, N_CLASSES
from integrations.continual_learning.cl4a_h1_measurement_contract import COUNTERS

SOURCE_SCOPE = "isolated_source_loop_probe_not_model_execution_or_hardware_counters"


def source_loop_probe(activity: tuple[int, ...], *, sparse: bool) -> dict:
    """Count loop events by visiting actual indices, never by multiplying a formula."""
    if type(sparse) is not bool:
        raise ValueError("sparse flag must be boolean")
    if not isinstance(activity, tuple) or len(activity) != INPUT_SIZE or any(
        type(bit) is not int or bit not in (0, 1) for bit in activity
    ):
        raise ValueError("expected 16 binary input bits")
    count = dict.fromkeys(COUNTERS, 0)
    active = []
    if sparse:
        # Mirrors the presynaptic activity scan, without creating model input or weights.
        for i, bit in enumerate(activity):
            count["input_visits"] += 1
            if bit:
                active.append(i)
                count["index_writes"] += 1
    for _hidden in range(HIDDEN_SIZE):
        if sparse:
            for _i in active:
                count["index_reads"] += 1
                count["forward_weight_terms"] += 1
                count["parameter_reads"] += 1
        else:
            for i in range(INPUT_SIZE):
                _ = activity[i]
                count["input_visits"] += 1
                count["forward_weight_terms"] += 1
                count["parameter_reads"] += 1
        # One hidden nonlinearity per neuron, counted as a planned source call,
        # not as measured physical CPU instructions.
        count["nonlinearity_counts"] += 1
    for _out in range(N_CLASSES):
        for _hidden in range(HIDDEN_SIZE):
            count["forward_weight_terms"] += 1
            count["parameter_reads"] += 1
    return {
        "kind": "CL4B_SOURCE_LOOP_EVENTS_ONLY",
        "scope": SOURCE_SCOPE,
        "arm": "sparse_local_hybrid" if sparse else "local_hybrid",
        "active_inputs": len(active) if sparse else sum(activity),
        "counters": count,
        "model_executed": False,
        "training_executed": False,
        "inference_executed": False,
        "energy_joules": None,
        "cpu_instructions": None,
        "hardware_memory_bytes": None,
    }
