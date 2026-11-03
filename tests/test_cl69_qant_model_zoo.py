"""CL-69: zoo layers — gradients, exact paths and agreement with earlier simulators."""
import unittest

import numpy as np

import experiments.cl69_qant_model_zoo.run as R


def xent(z, y):
    z = z - z.max(1, keepdims=True)
    return float(-(z[np.arange(len(y)), y] - np.log(np.exp(z).sum(1))).mean())


class CL69(unittest.TestCase):
    def test_hybrid_gradients_match_finite_differences(self):
        for spec, act in (([("lin", 8), ("kan", 10, 2)], "relu"), ([("kan", 6, 2), ("lin", 10)], None),
                          ([("lin", 8), ("lin", 10)], "sigmoid")):
            net = R.Net(1, spec, act)
            for L in net.layers:
                for k in L.p:
                    L.p[k] = L.p[k].astype(np.float64)
            rng = np.random.default_rng(0)
            X = rng.random((5, 784))
            y = rng.integers(0, 10, 5)
            # analytic gradients via one backward pass
            caches, acts, z = [], [], X
            for i, L in enumerate(net.layers):
                z, c = L.forward(z)
                caches.append(c)
                if i < len(net.layers) - 1 and act:
                    z = R.act_fwd(act, z)
                acts.append(z)
            zz = z - z.max(1, keepdims=True)
            p = np.exp(zz)
            p /= p.sum(1, keepdims=True)
            p[np.arange(5), y] -= 1
            g = p / 5
            grads = [None] * len(net.layers)
            for i in range(len(net.layers) - 1, -1, -1):
                if i < len(net.layers) - 1 and act:
                    g = R.act_bwd(act, acts[i], g)
                g, grads[i] = net.layers[i].backward(g, caches[i])
            for i, L in enumerate(net.layers):
                k = next(iter(L.p))
                idx = tuple(0 for _ in L.p[k].shape)
                eps = 1e-6
                L.p[k][idx] += eps
                up = xent(net.logits(X), y)
                L.p[k][idx] -= 2 * eps
                down = xent(net.logits(X), y)
                L.p[k][idx] += eps
                self.assertAlmostEqual(grads[i][k][idx], (up - down) / (2 * eps), places=6, msg=(spec, i, k))

    def test_kan_qant_matches_cl62_seq(self):
        net = R.Net(2, [("kan", 10, 4)], None)
        model = R.cl58.QFourier(2, 4)
        model.A, model.phi, model.b = net.layers[0].p["A"], net.layers[0].p["phi"], net.layers[0].p["b"]
        X = np.random.default_rng(3).random((4, 784)).astype(np.float32)
        np.testing.assert_array_equal(net.qant_logits(X), R.cl65.cl63.cl62.variant_logits(model, X, "seq"))

    def test_split_calls_and_counts(self):
        net = R.Net(4, [("kan", 10, 4)], None)
        self.assertEqual(net.longest_seq_sum(), 3136)
        self.assertEqual(net.longest_seq_sum(16), 196)
        self.assertEqual(net.qant_calls(16), 16)
        lin = R.Net(4, [("lin", 64), ("lin", 10)], "relu")
        self.assertEqual(lin.longest_seq_sum(), 0)

    def test_spearman_ties_are_order_independent(self):
        a = [784, 1568, 3136, 6272, 196, 112, 512, 3136]
        b = [162.3, 258.0, 337.0, 470.7, 42.7, 45.7, 95.0, 129.7]
        perm = [7, 0, 1, 3, 6, 5, 2, 4]
        r1 = R.spearman(a, b)
        r2 = R.spearman([a[i] for i in perm], [b[i] for i in perm])
        self.assertAlmostEqual(r1, r2, places=12)
        np.testing.assert_array_equal(R.average_ranks([3, 1, 3]), [2.5, 1, 2.5])


if __name__ == "__main__":
    unittest.main()
