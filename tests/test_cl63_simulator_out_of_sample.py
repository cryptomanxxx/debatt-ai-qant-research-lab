"""CL-63: Exp021 regeneration and grouped simulation are consistent."""
import unittest

import numpy as np

import experiments.cl63_simulator_out_of_sample.run as R


class CL63(unittest.TestCase):
    def test_exp021_inputs_deterministic_and_shaped(self):
        x1, p1, a1, k1, e1 = R.exp021_inputs(11, 4, 2)
        x2, *_ , e2 = R.exp021_inputs(11, 4, 2)
        np.testing.assert_array_equal(x1, x2)
        np.testing.assert_array_equal(e1, e2)
        self.assertEqual(e1.shape, (256, 10))
        self.assertEqual(p1.shape, (10, 4, 2))

    def test_simulators_close_for_single_term(self):
        x, p, a, k, exact = R.exp021_inputs(11, 1, 1)
        outs = [R.kan_sim(x, p, a, k, v) for v in R.VARIANTS_A]
        for o in outs[1:]:
            np.testing.assert_array_equal(o, outs[0])  # one term: every scheme is the same rounding
        self.assertLess(np.abs(outs[0] - exact).mean(), 0.01)

    def test_seq_orders_differ_only_in_order(self):
        x, p, a, k, _ = R.exp021_inputs(22, 16, 4)
        s1 = R.kan_sim(x, p, a, k, "seq")
        s2 = R.kan_sim(x, p, a, k, "seq_kmajor")
        self.assertLess(np.abs(s1 - s2).mean(), 0.05)

    def test_grouped_one_group_equals_monolithic_seq(self):
        model = R.cl58.QFourier(1, 4)
        X = np.random.default_rng(0).random((4, R.cl58.D)).astype(np.float32)
        g1 = R.grouped_logits(model, X, 1, "seq")
        mono = R.cl62.variant_logits(model, X, "seq")
        np.testing.assert_array_equal(g1, mono)


if __name__ == "__main__":
    unittest.main()
