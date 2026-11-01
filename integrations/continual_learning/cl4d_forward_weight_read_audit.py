"""CL-4D scoped unit-forward read tracing; never trains or runs research tasks.

This is an intentionally small correctness fixture. It counts Python __getitem__
calls on W1/W2 rows during one fixed-input _forward(), not hardware accesses.
No labels, datasets, optimizer steps, task sequences or energy measurements.
"""
from __future__ import annotations

from dataclasses import dataclass

from integrations.continual_learning.cl2_mlp_baseline import TinyDenseMLP
from integrations.continual_learning.cl2_sparse_forward import SparseForwardLocalMLP
from integrations.continual_learning.cl2_mlp_baseline import INPUT_SIZE

ARM_CLASSES = {"local_hybrid": TinyDenseMLP, "sparse_local_hybrid": SparseForwardLocalMLP}


class _ReadOnlyRow:
    """Transparent indexed float reads for the CL-2 forward kernels only."""
    def __init__(self, data: list[float], count: list[int]):
        self._data = data
        self._count = count

    def __getitem__(self, index: int) -> float:
        if type(index) is not int:
            raise ValueError("only scalar weight indexing belongs to this probe")
        self._count[0] += 1
        return self._data[index]


@dataclass(frozen=True)
class ForwardReadAudit:
    arm: str
    active_inputs: int
    w1_reads: int
    w2_reads: int
    logits: tuple[float, ...]
    scope: str = "one_synthetic_unit_forward_python_weight_row_getitem_only"
    training_executed: bool = False
    task_inference_executed: bool = False
    energy_joules: None = None
    hardware_memory_bytes: None = None


def audit_single_forward(arm: str, features: tuple[int, ...], *, seed: int = 7) -> ForwardReadAudit:
    """Run ONE fixed-input unit forward on fresh, untrained weights; no learning."""
    if arm not in ARM_CLASSES or type(seed) is not int or seed < 0:
        raise ValueError("arm/seed invalid")
    if not isinstance(features, tuple) or len(features) != INPUT_SIZE or any(
        type(v) is not int or v not in (0, 1) for v in features
    ):
        raise ValueError("expected exactly sixteen binary integer input values")
    model = ARM_CLASSES[arm](seed=seed)
    initial_steps = model.optimizer_steps
    original_w1, original_w2 = model.w1, model.w2
    w1_reads, w2_reads = [0], [0]
    try:
        model.w1 = [_ReadOnlyRow(row, w1_reads) for row in original_w1]
        model.w2 = [_ReadOnlyRow(row, w2_reads) for row in original_w2]
        # Internal unit-forward only. Not a heldout evaluation or study inference.
        _hidden, logits = model._forward(features)
    finally:
        model.w1, model.w2 = original_w1, original_w2
    if model.optimizer_steps != initial_steps:
        raise ValueError("forward fixture performed an unexpected optimizer step")
    return ForwardReadAudit(
        arm=arm, active_inputs=sum(features), w1_reads=w1_reads[0],
        w2_reads=w2_reads[0], logits=tuple(logits),
    )
