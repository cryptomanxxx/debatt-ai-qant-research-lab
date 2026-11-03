"""CL-62: accumulation schemes agree with float sums where they should."""
import unittest

import numpy as np

import experiments.cl62_qant_accumulation_simulator.run as R


class CL62Accumulation(unittest.TestCase):
    def test_exact_for_bf16_representable_small_integers(self):
        t = np.ones((2, 3, 100), np.float32)
        for mode in ("seq", "pair", "block32", "block256"):
            np.testing.assert_array_equal(R.accumulate(t, mode), 100.0, err_msg=mode)

    def test_seq_stagnates(self):
        # 256 + 1 is not representable in bfloat16 (8-bit significand), so a sequential sum stalls at 256.
        t = np.ones((1, 1, 300), np.float32)
        self.assertEqual(R.accumulate(t, "seq")[0, 0], 256.0)
        self.assertEqual(R.accumulate(t, "block32")[0, 0], 300.0)

    def test_close_to_float_sum_on_random_terms(self):
        rng = np.random.default_rng(0)
        t = R.bf16(rng.normal(0, 0.02, (4, 10, 3136)).astype(np.float32))
        exact = t.astype(np.float64).sum(-1)
        for mode, tol in (("pair", 0.05), ("block32", 0.02), ("block256", 0.02)):
            self.assertLess(np.abs(R.accumulate(t, mode) - exact).max(), tol, mode)

    def test_variant_logits_shapes(self):
        model = R.cl58.QFourier(1, 4)
        X = np.random.default_rng(1).random((5, R.cl58.D)).astype(np.float32)
        for v in R.VARIANTS:
            self.assertEqual(R.variant_logits(model, X, v).shape, (5, R.cl58.C), v)


if __name__ == "__main__":
    unittest.main()
