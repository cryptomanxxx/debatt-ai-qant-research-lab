"""CL-61: BPTT gradients with the spike-rate penalty (smooth spike stand-in)."""
import unittest

import numpy as np

import experiments.cl61_spike_rate_penalty.run as R


class CL61Gradients(unittest.TestCase):
    def check(self, H, lam):
        rng = np.random.default_rng(0)
        X = rng.random((8, R.D))
        X[X < 0.5] = 0
        y = rng.integers(0, R.C, 8)
        model = R.PenalizedSNN(1, H, lam, T=4, smooth=True)
        for n in model.p:
            model.p[n] = model.p[n].astype(np.float64)
        S = R.cl60.latency_encode(X, 4).astype(np.float64)
        analytic = model.grads(S, y)
        for name, idx in (("W1", (3, 200)), ("W1", (10, 400)), ("b1", (5,)), ("W2", (2, 7)), ("b2", (4,))):
            arr, eps = model.p[name], 1e-6
            arr[idx] += eps
            up = model.loss(S, y)
            arr[idx] -= 2 * eps
            down = model.loss(S, y)
            arr[idx] += eps
            self.assertAlmostEqual(analytic[name][idx], (up - down) / (2 * eps), places=6, msg=(H, lam, name, idx))

    def test_gradients_without_penalty(self):
        self.check(64, 0.0)

    def test_gradients_with_penalty(self):
        self.check(32, 3.0)

    def test_predict_counts(self):
        rng = np.random.default_rng(2)
        X = rng.random((10, R.D)).astype(np.float32)
        X[X < 0.5] = 0
        pred, (inp, hid) = R.PenalizedSNN(3, 64, 0.1).predict(X)
        self.assertEqual(pred.shape, (10,))
        self.assertEqual(inp, (X > 0).sum())
        self.assertLessEqual(hid, 10 * 64 * R.T)


if __name__ == "__main__":
    unittest.main()
