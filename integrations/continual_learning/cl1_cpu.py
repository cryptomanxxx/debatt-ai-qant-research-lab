"""CL-1 exploratory CPU-only sparse local-learning primitives.

This module is *not* Lava, Loihi, Q.ANT hardware, or an energy simulator.
No experiment is run at import time. See README before interpreting counters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class Counters:
    events: int = 0
    synaptic_reads: int = 0
    weight_writes: int = 0

    def __add__(self, other: "Counters") -> "Counters":
        return Counters(
            self.events + other.events,
            self.synaptic_reads + other.synaptic_reads,
            self.weight_writes + other.weight_writes,
        )


def encode_spikes(features: Iterable[float], threshold: float = 0.5) -> tuple[int, ...]:
    """Deterministic one-step threshold encoder; input magnitudes in [0, 1]."""
    if not 0 < threshold <= 1:
        raise ValueError("threshold must be in (0, 1]")
    values = tuple(float(x) for x in features)
    if not values or any(not 0 <= v <= 1 for v in values):
        raise ValueError("features must be nonempty and within [0, 1]")
    return tuple(int(v >= threshold) for v in values)


class LocalSpikeClassifier:
    """Minimal supervised three-factor-like *toy* rule, not Loihi STDP.

    Only active input synapses are read for inference and locally changed for
    the target class. The target label is an explicit external learning signal.
    """

    def __init__(self, input_size: int, n_classes: int, learning_rate: float = 0.1):
        if input_size <= 0 or n_classes < 2 or not 0 < learning_rate <= 1:
            raise ValueError("invalid classifier dimensions or learning rate")
        self.input_size = input_size
        self.n_classes = n_classes
        self.learning_rate = learning_rate
        self.weights = [[0.0] * input_size for _ in range(n_classes)]
        self.counters = Counters()

    def _active(self, spikes: Iterable[int]) -> list[int]:
        vector = tuple(spikes)
        if len(vector) != self.input_size or any(x not in (0, 1) for x in vector):
            raise ValueError("spikes must be a binary vector of input_size")
        return [i for i, spike in enumerate(vector) if spike]

    def predict(self, spikes: Iterable[int]) -> int:
        active = self._active(spikes)
        scores = [sum(row[i] for i in active) for row in self.weights]
        self.counters += Counters(
            events=len(active),
            synaptic_reads=len(active) * self.n_classes,
        )
        return max(range(self.n_classes), key=lambda k: (scores[k], -k))

    def learn(self, spikes: Iterable[int], label: int) -> None:
        if not isinstance(label, int) or not 0 <= label < self.n_classes:
            raise ValueError("invalid label")
        active = self._active(spikes)
        for i in active:
            self.weights[label][i] = min(1.0, self.weights[label][i] + self.learning_rate)
        self.counters += Counters(
            events=len(active),
            synaptic_reads=len(active),
            weight_writes=len(active),
        )

    def predict_then_learn(self, spikes: Iterable[int], label: int) -> int:
        vector = tuple(spikes)
        prediction = self.predict(vector)
        self.learn(vector, label)
        return prediction
