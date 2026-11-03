"""CL-59: BPTT gradients of the LIF network (smooth spike stand-in) and the dense net."""
import unittest

import numpy as np

import experiments.cl59_spiking_mnist.run as R


def xent(logits, y):
    z = logits - logits.max(1, keepdims=True)
    return -(z[np.arange(len(y)), y] - np.log(np.exp(z).sum(1))).mean()


class CL59Gradients(unittest.TestCase):
    def check(self, params, analytic, loss_fn):
        for name, idx in (("W1", (3, 200)), ("W1", (10, 400)), ("b1", (5,)), ("W2", (2, 7)), ("b2", (4,))):
            arr, eps = params[name], 1e-6
            arr[idx] += eps
            up = loss_fn()
            arr[idx] -= 2 * eps
            down = loss_fn()
            arr[idx] += eps
            self.assertAlmostEqual(analytic[name][idx], (up - down) / (2 * eps), places=6, msg=(name, idx))

    def test_snn_bptt_matches_finite_differences_with_smooth_spikes(self):
        rng = np.random.default_rng(0)
        X = rng.random((8, R.D))
        y = rng.integers(0, R.C, 8)
        model = R.SNN(1, T=5, smooth=True)
        for n in model.p:
            model.p[n] = model.p[n].astype(np.float64)
        S = (rng.random((5,) + X.shape) < X).astype(np.float64)
        self.check(model.p, model.grads(S, y), lambda: xent(model.run(S)[3], y))

    def test_ann_gradients(self):
        rng = np.random.default_rng(1)
        X = rng.random((8, R.D))
        y = rng.integers(0, R.C, 8)
        model = R.ANN(2)
        for n in model.p:
            model.p[n] = model.p[n].astype(np.float64)
        self.check(model.p, model.grads(X, y), lambda: xent(model.logits(X)[1], y))

    def test_snn_counts_spikes(self):
        rng = np.random.default_rng(3)
        X = rng.random((20, R.D)).astype(np.float32)
        pred, (inp, hid) = R.SNN(4, T=3).predict(X, np.random.default_rng(0))
        self.assertEqual(pred.shape, (20,))
        self.assertAlmostEqual(inp / (20 * 3), X.sum(1).mean(), delta=0.05 * X.sum(1).mean())
        self.assertGreaterEqual(hid, 0)


if __name__ == "__main__":
    unittest.main()
