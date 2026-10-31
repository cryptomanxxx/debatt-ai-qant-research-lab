"""Share active indices across sparse forward and local weight update.

Engineering ablation only. No training experiment, hardware or power claims.
"""
from __future__ import annotations

import math

from integrations.continual_learning.cl2_mlp_baseline import HIDDEN_SIZE, N_CLASSES
from integrations.continual_learning.cl2_sparse_forward import SparseForwardLocalMLP


class SharedIndexSparseForwardLocalMLP(SparseForwardLocalMLP):
    """Reuse the active list prepared once by LocalHebbianMLP.step.

    Standalone predict() still scans once through inherited _forward().
    This intervention changes only the index construction inside step(), not
    the weights, local update rule or temporal neuron dynamics.
    """

    def _forward_with_active(
        self, x: tuple[int, ...], active: list[int]
    ) -> tuple[list[float], list[float]]:
        hidden = [
            math.tanh(self.b1[j] + sum(self.w1[j][i] for i in active))
            for j in range(HIDDEN_SIZE)
        ]
        logits = [
            self.b2[k] + sum(self.w2[k][j] * hidden[j] for j in range(HIDDEN_SIZE))
            for k in range(N_CLASSES)
        ]
        return hidden, logits
