"""CL-55 model and metric checks on random data (no MNIST, no network)."""
import unittest

import numpy as np

import experiments.cl55_h2_permuted_mnist_forgetting.run as R


class CL55Checks(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.X = rng.random((128, R.D)).astype(np.float32)
        self.y = rng.integers(0, R.C, 128)

    def test_all_rules_train_without_nan(self):
        for cfg in (dict(H=32), dict(H=64, k=8), dict(H=64, k=8, rule="frozen"),
                    dict(H=32, rule="dfa"), dict(H=32, ewc=10000.0)):
            net = R.Net(1, **cfg)
            if cfg.get("ewc"):
                net.consolidate(self.X, self.y)
            for _ in range(5):
                net.train_batch(self.X, self.y)
            for name in R.PARAMS:
                self.assertTrue(np.isfinite(getattr(net, name)).all(), (cfg, name))

    def test_frozen_rule_keeps_hidden_layer_fixed(self):
        net = R.Net(2, H=64, k=8, rule="frozen")
        w1 = net.W1.copy()
        net.train_batch(self.X, self.y)
        self.assertTrue(np.array_equal(w1, net.W1))

    def test_strong_ewc_keeps_weights_near_anchor(self):
        free, anchored = R.Net(3, H=32), R.Net(3, H=32, ewc=1e6)
        anchored.consolidate(self.X, self.y)
        start = anchored.W2.copy()
        other = (self.y + 1) % R.C
        for _ in range(20):
            free.train_batch(self.X, other)
            anchored.train_batch(self.X, other)
        self.assertLess(np.abs(anchored.W2 - start).max(), np.abs(free.W2 - start).max())

    def test_metrics(self):
        m = R.metrics([[0.9, 0.1], [0.7, 0.8]])
        self.assertAlmostEqual(m["ACC"], 0.75)
        self.assertAlmostEqual(m["BWT"], -0.2)
        self.assertAlmostEqual(m["LA"], 0.85)


if __name__ == "__main__":
    unittest.main()
