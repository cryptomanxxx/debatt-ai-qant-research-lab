"""CL-60: latency encoder places exactly one spike per nonzero pixel at the right step."""
import unittest

import numpy as np

import experiments.cl60_latency_coded_snn.run as R


class CL60Encoding(unittest.TestCase):
    def test_one_spike_per_nonzero_pixel_at_expected_step(self):
        rng = np.random.default_rng(0)
        X = rng.random((6, R.D)).astype(np.float32)
        X[X < 0.4] = 0
        for T in (1, 4, 16):
            S = R.latency_encode(X, T)
            self.assertEqual(S.shape, (T,) + X.shape)
            np.testing.assert_array_equal(S.sum(0), (X > 0).astype(np.float32))
            t = S.argmax(0)
            expected = np.rint((1 - X) * (T - 1))
            np.testing.assert_array_equal(t[X > 0], expected[X > 0])

    def test_extremes(self):
        X = np.array([[1.0, 0.0, 1e-3]], np.float32)
        S = R.latency_encode(X, 8)
        self.assertEqual(S[0, 0, 0], 1)  # brightest spikes first
        self.assertEqual(S[:, 0, 1].sum(), 0)  # zero pixels never spike
        self.assertEqual(S[7, 0, 2], 1)  # faintest spikes last

    def test_predict_counts_input_spikes(self):
        rng = np.random.default_rng(1)
        X = rng.random((10, R.D)).astype(np.float32)
        X[X < 0.5] = 0
        pred, (inp, hid) = R.LatencySNN(3, T=4).predict(X)
        self.assertEqual(pred.shape, (10,))
        self.assertEqual(inp, (X > 0).sum())
        self.assertGreaterEqual(hid, 0)


if __name__ == "__main__":
    unittest.main()
