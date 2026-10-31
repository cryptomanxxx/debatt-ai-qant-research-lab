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

    def step(self, features: Iterable[int], label: int) -> float:
        """One optional local-rule update, not a dataset training runner."""
        if type(label) is not int or not 0 <= label < N_CLASSES:
            raise ValueError("label must be 0 or 1")
        x = self._input(features)
        active = [i for i, bit in enumerate(x) if bit]
        hidden, logits = self._forward(x)
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

        self.logical_counters += Counters(
            events=len(active),
            synaptic_reads=len(active) * HIDDEN_SIZE + HIDDEN_SIZE * N_CLASSES,
            weight_writes=(
                len(active) * HIDDEN_SIZE
                + HIDDEN_SIZE  # hidden biases
                + HIDDEN_SIZE * N_CLASSES  # output weights
                + N_CLASSES  # output biases
            ),
        )
        self.optimizer_steps += 1
        return loss
