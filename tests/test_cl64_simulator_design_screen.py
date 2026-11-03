"""CL-64: hierarchical NumPy model gradients and simulated calls."""
import unittest

import numpy as np

import experiments.cl64_simulator_design_screen.run as R


def xent(logits, y):
    z = logits - logits.max(1, keepdims=True)
    return -(z[np.arange(len(y)), y] - np.log(np.exp(z).sum(1))).mean()


class CL64(unittest.TestCase):
    def test_hier_gradients_match_finite_differences(self):
        rng = np.random.default_rng(0)
        model = R.Hier(1, 16, 2)
        for k in model.p:
            model.p[k] = model.p[k].astype(np.float64)
        X = rng.random((6, R.cl58.D))
        y = rng.integers(0, 10, 6)
        analytic = model.grads(X, y)
        for name, idx in (("s1_A", (3, 1, 10, 2)), ("s1_phi", (5, 0, 7, 1)), ("s2_A", (4, 9, 3)), ("s2_phi", (2, 20, 0)), ("s1_b", (7, 1)), ("s2_b", (3,))):
            arr, eps = model.p[name], 1e-6
            arr[idx] += eps
            up = xent(model.logits(X), y)
            arr[idx] -= 2 * eps
            down = xent(model.logits(X), y)
            arr[idx] += eps
            self.assertAlmostEqual(analytic[name][idx], (up - down) / (2 * eps), places=6, msg=name)

    def test_exact_call_matches_model_logits(self):
        model = R.Hier(2, 8, 2)
        X = np.random.default_rng(1).random((4, R.cl58.D)).astype(np.float32)
        np.testing.assert_allclose(model.sim_logits(X, "exact", "exact"), model.logits(X), rtol=1e-5, atol=1e-5)

    def test_seq_call_is_bf16(self):
        model = R.Hier(3, 28, 2)
        X = np.random.default_rng(2).random((3, R.cl58.D)).astype(np.float32)
        out = model.sim_logits(X, "seq", "seq")
        np.testing.assert_array_equal(out, R.bf16(out))


if __name__ == "__main__":
    unittest.main()
