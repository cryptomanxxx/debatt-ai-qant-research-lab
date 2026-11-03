"""CL-67: the C sequential-bf16 kernel is bit-identical to the NumPy simulator."""
import unittest

import numpy as np

import experiments.cl67_accumulation_aware_training.run as R


class CL67(unittest.TestCase):
    def test_c_kernel_matches_numpy(self):
        rng = np.random.default_rng(0)
        for n in (1, 7, 300, 3136):
            t = R.bf16(rng.normal(0, 0.05, (5, 10, n)).astype(np.float32))
            np.testing.assert_array_equal(R.seq_sum(t), R.cl62.accumulate(t, "seq"), err_msg=str(n))

    def test_sim_logits_match_cl62(self):
        model = R.AccModel(1, 4)
        X = np.random.default_rng(1).random((6, R.cl58.D)).astype(np.float32)
        np.testing.assert_array_equal(R.sim_logits(model, X), R.cl62.variant_logits(model, X, "seq"))

    def test_seq_ste_step_changes_parameters(self):
        model = R.AccModel(2, 4)
        X = np.random.default_rng(2).random((16, R.cl58.D)).astype(np.float32)
        y = np.arange(16) % 10
        A0 = model.A.copy()
        model.train_seq_ste(X, y)
        self.assertGreater(np.abs(model.A - A0).max(), 0)


if __name__ == "__main__":
    unittest.main()
