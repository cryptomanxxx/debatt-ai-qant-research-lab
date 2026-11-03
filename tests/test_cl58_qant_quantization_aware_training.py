"""CL-58: bfloat16 simulator and KAN-layer gradients (random data, no download)."""
import copy
import unittest

import numpy as np

import experiments.cl58_qant_quantization_aware_training.run as R


class CL58Checks(unittest.TestCase):
    def test_bf16_matches_ml_dtypes_when_available(self):
        rng = np.random.default_rng(0)
        x = (rng.standard_normal(100000) * 10.0 ** rng.integers(-6, 6, 100000)).astype(np.float32)
        x[:4] = [0.0, -0.0, 1.0, -2.5]
        try:
            from ml_dtypes import bfloat16
        except ImportError:
            # Without ml_dtypes, check basic properties: idempotent and within 2^-8 relative error.
            y = R.bf16(x)
            self.assertTrue(np.array_equal(R.bf16(y), y))
            self.assertTrue(np.all(np.abs(y - x) <= np.abs(x) * 2.0 ** -8))
            return
        self.assertTrue(np.array_equal(R.bf16(x), x.astype(bfloat16).astype(np.float32)))

    def test_exact_cosine_gradients_match_finite_differences(self):
        rng = np.random.default_rng(1)
        X = rng.random((6, R.D)).astype(np.float32)
        y = rng.integers(0, R.C, 6)
        model = R.QFourier(3, 2)
        for name in ("A", "phi", "b"):
            setattr(model, name, getattr(model, name).astype(np.float64))

        def loss(m):
            z = m.logits(X.astype(np.float64))
            z = z - z.max(1, keepdims=True)
            return -(z[np.arange(6), y] - np.log(np.exp(z).sum(1))).mean()

        # Adam's first step moves every parameter by exactly -lr * sign(gradient).
        stepped = copy.deepcopy(model)
        stepped.train_batch(X, y, "exact_cosine", None)
        for name, idx in (("A", (2, 300, 1)), ("phi", (5, 100, 0)), ("b", (7,))):
            arr, eps = getattr(model, name), 1e-6
            arr[idx] += eps
            up = loss(model)
            arr[idx] -= 2 * eps
            down = loss(model)
            arr[idx] += eps
            numeric = (up - down) / (2 * eps)
            moved = getattr(stepped, name)[idx] - getattr(model, name)[idx]
            if abs(numeric) > 1e-7:
                self.assertEqual(np.sign(moved), -np.sign(numeric), (name, idx))
                self.assertAlmostEqual(abs(moved), R.LR, delta=R.LR * 1e-3)

    def test_training_runs_for_all_arms(self):
        rng = np.random.default_rng(2)
        X = rng.random((32, R.D)).astype(np.float32)
        y = rng.integers(0, R.C, 32)
        for arm in R.ARMS:
            m = R.QFourier(4, 2)
            m.train_batch(X, y, arm, np.random.default_rng(0))
            self.assertTrue(all(np.isfinite(getattr(m, n)).all() for n in ("A", "phi", "b")))


if __name__ == "__main__":
    unittest.main()
