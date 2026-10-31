"""Capacity-matched CL-2 local-rule candidate: NOT a temporal SNN or experiment.

Architecture, parameter count and initial weight distribution match TinyDenseMLP.
The hidden layer uses a local Hebbian-like update without a propagated label
gradient; the output layer uses an externally supervised local readout signal.
"""
from __future__ import annotations

import math
from typing import Iterable

from integrations.continual_learning.cl1_cpu import Counters
from integrations.continual_learning.cl2_mlp_baseline import (
    HIDDEN_SIZE,
    INPUT_SIZE,
    N_CLASSES,
    TinyDenseMLP,
)


class LocalHebbianMLP(TinyDenseMLP):
    """Sparse presynaptic traversal, local hidden updates, supervised readout.

    The hidden rule depends only on the current presynaptic input bit and
    postsynaptic hidden activity. Output updates use the globally provided
    target label and softmax error; the full algorithm is NOT biologically
    local, and no temporal spike dynamics are modeled.
    """

    def __init__(
        self, seed: int = 0, learning_rate: float = 0.05, local_rate: float = 0.01
    ):
        super().__init__(seed=seed, learning_rate=learning_rate)
        if (
            isinstance(local_rate, bool)
            or not math.isfinite(local_rate)
            or not 0 < local_rate <= 1
        ):
            raise ValueError("local_rate must be finite and in (0, 1]")
        self.local_rate = float(local_rate)
        self.logical_counters = Counters()

    def _forward_first_layer_reads(self, active_count: int) -> int:
        """Source-level weight traversal count; dense by default."""
        return INPUT_SIZE * HIDDEN_SIZE

    def _forward_with_active(
        self, x: tuple[int, ...], active: list[int]
    ) -> tuple[list[float], list[float]]:
        """Hook: dense implementation ignores precomputed indices."""
        return self._forward(x)

    def step(self, features: Iterable[int], label: int) -> float:
        """One optional local-rule update, not a dataset training runner."""
        if type(label) is not int or not 0 <= label < N_CLASSES:
            raise ValueError("label must be 0 or 1")
        x = self._input(features)
        active = [i for i, bit in enumerate(x) if bit]
        hidden, logits = self._forward_with_active(x, active)
        offset = max(logits)
        scaled = [math.exp(value - offset) for value in logits]
        denom = sum(scaled)
        probabilities = [value / denom for value in scaled]
        loss = -math.log(probabilities[label])

        # Output layer: local pre/post activity and explicit supervised error.
        # The error itself depends on both output scores; not pure STDP.
        for k in range(N_CLASSES):
            delta = int(k == label) - probabilities[k]
            for j in range(HIDDEN_SIZE):
                self.w2[k][j] += self.learning_rate * delta * hidden[j]
            self.b2[k] += self.learning_rate * delta

        # Hidden layer: Hebbian-like pre/post updates; NO label-gradient path.
        for j in range(HIDDEN_SIZE):
            for i in active:
                self.w1[j][i] += self.local_rate * hidden[j]
            self.b1[j] += self.local_rate * hidden[j]

        # Count ALL actual weight-loop accesses on a consistent narrow scope:
        # dense forward W1/W2 plus read-modify-write of updated W1/W2.
        # Biases and input scanning are tracked separately by the static
        # cl2_operation_accounting profile; this is not hardware telemetry.
        self.logical_counters += Counters(
            events=len(active),
            synaptic_reads=(
                self._forward_first_layer_reads(len(active))
                + HIDDEN_SIZE * N_CLASSES
                + HIDDEN_SIZE * N_CLASSES
                + len(active) * HIDDEN_SIZE
            ),
            weight_writes=HIDDEN_SIZE * N_CLASSES + len(active) * HIDDEN_SIZE,
        )
        self.optimizer_steps += 1
        return loss
