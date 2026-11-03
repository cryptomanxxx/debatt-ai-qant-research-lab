"""CL-68: C sequential-bf16 dot product and the MLP gradients."""
import unittest

import numpy as np

import experiments.cl68_linear_fprop_accumulation.run as R


class CL68(unittest.TestCase):
    def test_seq_dot_matches_numpy_reference(self):
        rng = np.random.default_rng(0)
        x = R.bf16(rng.random((3, 50)).astype(np.float32))
        w = R.bf16(rng.normal(0, 0.1, (4, 50)).astype(np.float32))
        ref = np.zeros((3, 4), np.float32)
        for b in range(3):
            for h in range(4):
                s = np.float32(0)
                for k in range(50):
                    s = R.bf16(np.array([s + R.bf16(np.array([x[b, k] * w[h, k]], np.float32))[0]], np.float32))[0]
                ref[b, h] = s
        np.testing.assert_array_equal(R.seq_dot(x, w), ref)

    def test_mlp_gradient_direction(self):
        rng = np.random.default_rng(1)
        X = rng.random((16, 784)).astype(np.float32)
        y = np.arange(16) % 10
        m = R.MLP(3, [32])

        def loss():
            z = m.forward(X)[-1]
            z = z - z.max(1, keepdims=True)
            return float(-(z[np.arange(16), y] - np.log(np.exp(z).sum(1))).mean())

        before = loss()
        for _ in range(20):
            m.train_batch(X, y)
        self.assertLess(loss(), before)

    def test_f32acc_path_close_to_reference(self):
        m = R.MLP(4, [64])
        X = np.random.default_rng(2).random((200, 784)).astype(np.float32)
        self.assertGreater((m.qant_path(X, "f32acc") == m.predict(X)).mean(), 0.9)


if __name__ == "__main__":
    unittest.main()
