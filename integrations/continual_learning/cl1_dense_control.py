"""Dense digital control for CL-1, aligned to the sparse toy classifier.

No external libraries, photonic runtime, model training, or execution on import.
Counters represent logical algorithmic accesses, NOT physical energy or bytes.
"""

from __future__ import annotations

from typing import Iterable

from integrations.continual_learning.cl1_cpu import Counters


class DenseControlClassifier:
    """Dense, supervised linear toy control with the same class-by-input weights.

    For each observation, score every input/class pair and update every target-
    class weight (including zero-valued feature contributions). This is an
    intentionally dense implementation of the sparse toy's update rule,
    *not* a backpropagation neural network or a hardware performance model.
    """

    def __init__(self, input_size: int, n_classes: int, learning_rate: float = 0.1):
        if input_size <= 0 or n_classes < 2 or not 0 < learning_rate <= 1:
            raise ValueError("invalid classifier dimensions or learning rate")
        self.input_size = input_size
        self.n_classes = n_classes
        self.learning_rate = learning_rate
        self.weights = [[0.0] * input_size for _ in range(n_classes)]
        self.counters = Counters()

    def _features(self, features: Iterable[int]) -> tuple[int, ...]:
        vector = tuple(features)
        if len(vector) != self.input_size or any(x not in (0, 1) for x in vector):
            raise ValueError("features must be a binary vector of input_size")
        return vector

    def predict(self, features: Iterable[int]) -> int:
        vector = self._features(features)
        scores = [
            sum(row[i] * vector[i] for i in range(self.input_size))
            for row in self.weights
        ]
        self.counters += Counters(
            events=sum(vector),
            synaptic_reads=self.n_classes * self.input_size,
        )
        return max(range(self.n_classes), key=lambda k: (scores[k], -k))

    def learn(self, features: Iterable[int], label: int) -> None:
        if not isinstance(label, int) or not 0 <= label < self.n_classes:
            raise ValueError("invalid label")
        vector = self._features(features)
        for i in range(self.input_size):
            self.weights[label][i] = min(
                1.0, self.weights[label][i] + self.learning_rate * vector[i]
            )
        self.counters += Counters(
            events=sum(vector),
            synaptic_reads=self.input_size,
            weight_writes=self.input_size,
        )

    def predict_then_learn(self, features: Iterable[int], label: int) -> int:
        vector = self._features(features)
        prediction = self.predict(vector)
        self.learn(vector, label)
        return prediction
