"""CL-53 model checks on random data (no MNIST download, no network)."""
import copy
import unittest

import numpy as np

import experiments.cl53_mnist_hidden_sparsity.run as R


class CL53ModelChecks(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.X = rng.random((32, R.D)).astype(np.float32)
        self.y = rng.integers(0, R.C, 32)

    def test_sparse_inference_path_matches_masked_path(self):
        for cfg in (dict(H=64), dict(H=128, k=8, sel="exact"),
                    dict(H=128, k=8, sel="lowrank", r=8, c=32)):
            net = R.Net(1, **cfg)
            for _ in range(3):
                net.train_batch(self.X, self.y)
            self.assertGreater((net.predict_sparse(self.X) == net.predict(self.X)).mean(), 0.99)

    def test_gradients_match_finite_differences(self):
        X, y = self.X[:8].astype(np.float64), self.y[:8]
        net = R.Net(2, H=32, k=8, sel="exact")
        for name in ("W1", "b1", "W2", "b2"):
            setattr(net, name, getattr(net, name).astype(np.float64))

        def loss():
            logits = net.forward(X)[3]
            logits = logits - logits.max(1, keepdims=True)
            p = np.exp(logits)
            p /= p.sum(1, keepdims=True)
            return -np.log(p[np.arange(len(y)), y]).mean()

        stepped = copy.deepcopy(net)
        old_lr, R.LR = R.LR, 1.0
        try:
            stepped.train_batch(X, y)
        finally:
            R.LR = old_lr
        for name, idx in (("W1", (5, 300)), ("b1", (3,)), ("W2", (1, 2)), ("b2", (4,))):
            analytic = getattr(net, name)[idx] - getattr(stepped, name)[idx]
            arr, eps = getattr(net, name), 1e-6
            arr[idx] += eps
            up = loss()
            arr[idx] -= 2 * eps
            down = loss()
            arr[idx] += eps
            self.assertAlmostEqual(analytic, (up - down) / (2 * eps), places=6)


if __name__ == "__main__":
    unittest.main()
