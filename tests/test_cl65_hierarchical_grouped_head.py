"""CL-65: grouped head with one group equals the CL-64 single head call."""
import unittest

import numpy as np

import experiments.cl65_hierarchical_grouped_head.run as R


class CL65(unittest.TestCase):
    def test_one_group_equals_single_call(self):
        model = R.cl64.Hier(1, 28, 2)
        X = np.random.default_rng(0).random((4, R.cl58.D)).astype(np.float32)
        H = R.stage1(model, X, "seq")
        np.testing.assert_array_equal(R.grouped_head(model, H, 1, "seq"), model.sim_logits(X, "seq", "seq"))

    def test_exact_paths_match_model(self):
        model = R.cl64.Hier(2, 16, 2)
        X = np.random.default_rng(1).random((3, R.cl58.D)).astype(np.float32)
        out = R.grouped_head(model, R.stage1(model, X, "exact"), 4, "exact")
        np.testing.assert_allclose(out, model.logits(X), rtol=1e-5, atol=1e-5)

    def test_grouped_head_close_to_exact(self):
        model = R.cl64.Hier(3, 28, 2)
        X = np.random.default_rng(2).random((3, R.cl58.D)).astype(np.float32)
        H = R.stage1(model, X, "exact")
        exact = R.grouped_head(model, H, 1, "exact")
        self.assertLess(np.abs(R.grouped_head(model, H, 8, "seq") - exact).max(), 0.1)


if __name__ == "__main__":
    unittest.main()
