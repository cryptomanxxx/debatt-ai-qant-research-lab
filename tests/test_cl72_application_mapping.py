"""CL-72: workload accounting sanity checks and CNN gradients."""
import unittest

import numpy as np

import experiments.cl72_application_mapping.run as R


class CL72(unittest.TestCase):
    def test_resnet18_macs_match_published_value(self):
        s = R.resnet18().summary()
        self.assertAlmostEqual(s["total_macs"] / 1e9, 1.82, delta=0.05)  # ~1.8 GMACs, commonly cited
        self.assertAlmostEqual(s["weights_mb_bf16"] / 2, 11.7, delta=0.3)  # ~11.7M parameters

    def test_mobilenet_v2_macs_match_published_value(self):
        s = R.mobilenet_v2().summary()
        self.assertAlmostEqual(s["total_macs"] / 1e9, 0.30, delta=0.03)  # ~300 MMACs, commonly cited
        self.assertEqual(s["min_fan_in"], 9)

    def test_im2col_traffic_counts_expanded_patches(self):
        w = R.Workload("one conv", "")
        w.conv(24, 24, 1, 16, 5, 28, 28)
        layer = w.layers[0]
        self.assertEqual(layer["io_bytes"], 2 * (28 * 28 + 24 * 24 * 16))
        self.assertEqual(layer["io_bytes_matrix"], 2 * (24 * 24 * 25 + 24 * 24 * 16))

    def test_bert_static_macs(self):
        s = R.bert(128).summary()
        static = s["total_macs"] * s["static_mac_share"]
        self.assertAlmostEqual(static, 12 * 128 * (4 * 768 * 768 + 2 * 768 * 3072), delta=1)

    def test_im2col_col2im_adjoint(self):
        rng = np.random.default_rng(0)
        x = rng.random((2, 8, 8, 3)).astype(np.float32)
        g = rng.random((2, 4, 4, 75)).astype(np.float32)
        lhs = float((R.im2col(x, 5) * g).sum())
        rhs = float((x * R.col2im(g, 5, x.shape)).sum())
        self.assertAlmostEqual(lhs, rhs, places=3)

    def test_cnn_training_reduces_loss(self):
        rng = np.random.default_rng(1)
        X = rng.random((32, 784)).astype(np.float32)
        y = np.arange(32) % 10
        m = R.CNN(2)

        def loss():
            z = m.forward(X)[0]
            z = z - z.max(1, keepdims=True)
            return float(-(z[np.arange(32), y] - np.log(np.exp(z).sum(1))).mean())
        before = loss()
        for _ in range(30):
            m.train_batch(X, y)
        self.assertLess(loss(), before - 0.1)


if __name__ == "__main__":
    unittest.main()
