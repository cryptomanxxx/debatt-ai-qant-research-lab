"""Small deterministic nonlinear *feature-map* CPU control for CL-2.

The XOR interaction is hard-coded as four pair-indicator features. Only the
linear readout is trainable. This is NOT a learned-hidden-layer ANN, a Loihi
SNN, an energy simulator or an approved research experiment.
"""
from __future__ import annotations

from typing import Iterable

from integrations.continual_learning.cl1_cpu import Counters


INPUT_SIZE = 16
BASIS_SIZE = 4
FEATURE_SIZE = INPUT_SIZE + BASIS_SIZE
N_CLASSES = 2


def nonlinear_features(values: Iterable[int]) -> tuple[int, ...]:
    """Raw bits plus four mutually exclusive indicators for (bit0, bit1)."""
    bits = tuple(values)
    if len(bits) != INPUT_SIZE or any(type(bit) is not int or bit not in (0, 1) for bit in bits):
        raise ValueError("expected exactly sixteen binary integer features")
    index = 2 * bits[0] + bits[1]
    basis = tuple(int(k == index) for k in range(BASIS_SIZE))
    return bits + basis


class DenseNonlinearControl:
    """Dense two-output linear readout over fixed nonlinear features.

    Trainable parameters: 2 * (16 raw inputs + 4 XOR basis indicators) = 40,
    no bias. The fixed basis is an architectural prior and its construction
    cost is NOT included in Counters. This architecture is *not* matched in
    capacity or information processing to the current 8-/32-weight CL-1 toys.
    """

    def __init__(self, learning_rate: float = 0.1):
        if isinstance(learning_rate, bool) or not 0 < learning_rate <= 1:
            raise ValueError("learning_rate must be in (0, 1]")
        self.learning_rate = float(learning_rate)
        self.weights = [[0.0] * FEATURE_SIZE for _ in range(N_CLASSES)]
        self.counters = Counters()

    @property
    def parameter_count(self) -> int:
        return N_CLASSES * FEATURE_SIZE

    def predict(self, values: Iterable[int]) -> int:
        features = nonlinear_features(values)
        scores = [
            sum(row[i] * features[i] for i in range(FEATURE_SIZE))
            for row in self.weights
        ]
        self.counters += Counters(
            events=sum(features[:INPUT_SIZE]),
            synaptic_reads=N_CLASSES * FEATURE_SIZE,
        )
        return max(range(N_CLASSES), key=lambda k: (scores[k], -k))

    def learn(self, values: Iterable[int], label: int) -> None:
        if type(label) is not int or not 0 <= label < N_CLASSES:
            raise ValueError("label must be 0 or 1")
        features = nonlinear_features(values)
        for i, feature in enumerate(features):
            self.weights[label][i] = min(
                1.0, self.weights[label][i] + self.learning_rate * feature
            )
        self.counters += Counters(
            events=sum(features[:INPUT_SIZE]),
            synaptic_reads=FEATURE_SIZE,
            weight_writes=FEATURE_SIZE,
        )

    def predict_then_learn(self, values: Iterable[int], label: int) -> int:
        bits = tuple(values)
        guess = self.predict(bits)
        self.learn(bits, label)
        return guess
