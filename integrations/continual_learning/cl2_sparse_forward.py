"""Sparse-forward ablation for CL-2 local model; not a temporal SNN.

Only numerical equivalence and algorithmic traversal are evaluated in unit
contracts. No experiment runner, physical energy measurement or SDK access.
"""
from __future__ import annotations

import math

from integrations.continual_learning.cl2_local_mlp import LocalHebbianMLP
from integrations.continual_learning.cl2_mlp_baseline import HIDDEN_SIZE, N_CLASSES


class SparseForwardLocalMLP(LocalHebbianMLP):
    """Same weights and update rule; skip zero-valued W1 forward products.

    The active-input index list is rebuilt inside _forward; the inherited
    step also scans for active inputs once. Both scans contribute overhead.
    """

    def _forward_first_layer_reads(self, active_count: int) -> int:
        return active_count * HIDDEN_SIZE

    def _forward(self, x: tuple[int, ...]) -> tuple[list[float], list[float]]:
        active = [i for i, value in enumerate(x) if value]
        hidden = [
            math.tanh(self.b1[j] + sum(self.w1[j][i] for i in active))
            for j in range(HIDDEN_SIZE)
        ]
        logits = [
            self.b2[k] + sum(self.w2[k][j] * hidden[j] for j in range(HIDDEN_SIZE))
            for k in range(N_CLASSES)
        ]
        return hidden, logits
