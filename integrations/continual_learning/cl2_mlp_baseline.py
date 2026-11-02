"""Dependency-free dense 16->8->2 MLP with learned hidden features.

A software contract reference, not a CL-2 research run, SNN, photonic
processor, or energy estimate. No work occurs on module import.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Iterable

from integrations.continual_learning.cl10k_weight_update_hooks import note_weight_update
from integrations.continual_learning.cl10l_forward_hooks import attached_candidate, candidate_forward


INPUT_SIZE = 16
HIDDEN_SIZE = 8
N_CLASSES = 2
PARAMETER_COUNT = INPUT_SIZE * HIDDEN_SIZE + HIDDEN_SIZE + HIDDEN_SIZE * N_CLASSES + N_CLASSES


@dataclass(frozen=True)
class Gradients:
    w1: tuple[tuple[float, ...], ...]
    b1: tuple[float, ...]
    w2: tuple[tuple[float, ...], ...]
    b2: tuple[float, ...]


class TinyDenseMLP:
    """Fully trained-feature reference candidate: tanh hidden, softmax loss.

    Both layers and both bias vectors are trainable. Dense full-parameter SGD
    updates are intentionally explicit, permitting later instrumentation.
    """

    def __init__(self, seed: int = 0, learning_rate: float = 0.05):
        if type(seed) is not int or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if isinstance(learning_rate, bool) or not math.isfinite(learning_rate) or not 0 < learning_rate <= 1:
            raise ValueError("learning_rate must be finite and in (0, 1]")
        rng = random.Random(seed)
        self.w1 = [[rng.uniform(-0.2, 0.2) for _ in range(INPUT_SIZE)] for _ in range(HIDDEN_SIZE)]
        self.b1 = [0.0] * HIDDEN_SIZE
        self.w2 = [[rng.uniform(-0.2, 0.2) for _ in range(HIDDEN_SIZE)] for _ in range(N_CLASSES)]
        self.b2 = [0.0] * N_CLASSES
        self.learning_rate = float(learning_rate)
        self.optimizer_steps = 0

    @property
    def parameter_count(self) -> int:
        return PARAMETER_COUNT

    @staticmethod
    def _input(features: Iterable[int]) -> tuple[int, ...]:
        x = tuple(features)
        if len(x) != INPUT_SIZE or any(type(v) is not int or v not in (0, 1) for v in x):
            raise ValueError("expected exactly sixteen binary integer inputs")
        return x

    def _forward(self, x: tuple[int, ...]) -> tuple[list[float], list[float]]:
        if attached_candidate(self):
            return candidate_forward(self, x, mode="dense")
        hidden = [
            math.tanh(self.b1[j] + sum(self.w1[j][i] * x[i] for i in range(INPUT_SIZE)))
            for j in range(HIDDEN_SIZE)
        ]
        logits = [
            self.b2[k] + sum(self.w2[k][j] * hidden[j] for j in range(HIDDEN_SIZE))
            for k in range(N_CLASSES)
        ]
        return hidden, logits

    def predict(self, features: Iterable[int]) -> int:
        _, logits = self._forward(self._input(features))
        return max(range(N_CLASSES), key=lambda k: (logits[k], -k))

    def loss_and_gradients(self, features: Iterable[int], label: int) -> tuple[float, Gradients]:
        if type(label) is not int or not 0 <= label < N_CLASSES:
            raise ValueError("label must be 0 or 1")
        x = self._input(features)
        hidden, logits = self._forward(x)
        offset = max(logits)
        exp_scores = [math.exp(v - offset) for v in logits]
        denominator = sum(exp_scores)
        probs = [v / denominator for v in exp_scores]
        loss = -math.log(probs[label])
        delta_out = [probs[k] - int(k == label) for k in range(N_CLASSES)]
        gw2 = tuple(
            tuple(delta_out[k] * hidden[j] for j in range(HIDDEN_SIZE))
            for k in range(N_CLASSES)
        )
        gb2 = tuple(delta_out)
        delta_hidden = [
            (1 - hidden[j] * hidden[j])
            * sum(self.w2[k][j] * delta_out[k] for k in range(N_CLASSES))
            for j in range(HIDDEN_SIZE)
        ]
        gw1 = tuple(
            tuple(delta_hidden[j] * x[i] for i in range(INPUT_SIZE))
            for j in range(HIDDEN_SIZE)
        )
        return loss, Gradients(gw1, tuple(delta_hidden), gw2, gb2)

    def step(self, features: Iterable[int], label: int) -> float:
        """One explicit SGD update; never invoked by CI as a dataset experiment."""
        loss, grad = self.loss_and_gradients(features, label)
        rate = self.learning_rate
        for j in range(HIDDEN_SIZE):
            for i in range(INPUT_SIZE):
                note_weight_update(self, "w1_update_read")
                self.w1[j][i] -= rate * grad.w1[j][i]
                note_weight_update(self, "w1_update_write")
            self.b1[j] -= rate * grad.b1[j]
        for k in range(N_CLASSES):
            for j in range(HIDDEN_SIZE):
                note_weight_update(self, "w2_update_read")
                self.w2[k][j] -= rate * grad.w2[k][j]
                note_weight_update(self, "w2_update_write")
            self.b2[k] -= rate * grad.b2[k]
        self.optimizer_steps += 1
        return loss
