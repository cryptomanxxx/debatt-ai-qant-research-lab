"""CL-70: generators are deterministic and the Q.ANT code path runs end to end on a fake backend."""
import sys
import types
import unittest
from unittest import mock

import numpy as np

import experiments.cl70_fanin_budget.run as R


def fake_modules():
    ai = types.ModuleType("qant_native_computing_toolkit.ai")
    f = lambda v: np.asarray(v, np.float32)
    ai.linear_fprop = lambda x, w: R.bf16(f(x) @ f(w).T)
    ai.calc_kan_layer_fprop = lambda x, p, a, k: np.einsum("bcik,cik->bc", np.cos(f(x)[:, None, :, None] * f(k) + f(p)[None]), f(a))
    ai.relu_fprop = lambda v: np.maximum(f(v), 0)
    ai.sigmoid_fprop = lambda v: R.bf16(1 / (1 + np.exp(-f(v))))
    def softmax(v):
        e = np.exp(f(v) - f(v).max(1, keepdims=True))
        return R.bf16(e / e.sum(1, keepdims=True))
    ai.softmax_fprop = softmax
    ai.add_bias_fprop = lambda a, b: R.bf16(f(a) + f(b))
    pkg = types.ModuleType("qant_native_computing_toolkit")
    pkg.ai = ai
    md = types.ModuleType("ml_dtypes")
    md.bfloat16 = np.float32
    return {"qant_native_computing_toolkit": pkg, "qant_native_computing_toolkit.ai": ai, "ml_dtypes": md}


class CL70(unittest.TestCase):
    def test_inputs_deterministic(self):
        a = R.linear_inputs(11, 16)
        b = R.linear_inputs(11, 16)
        np.testing.assert_array_equal(a[0], b[0])
        k1 = R.kan_inputs(22, 4)
        k2 = R.kan_inputs(22, 4)
        np.testing.assert_array_equal(k1[4], k2[4])

    def test_qant_path_runs_on_fake_backend(self):
        with mock.patch.dict(sys.modules, fake_modules()), \
                mock.patch.object(R, "LINEAR_FANIN", (1, 16)), mock.patch.object(R, "KAN_INPUTS", (1, 4)), \
                mock.patch.object(R, "SEEDS", (11,)):
            rows, acts = R.run_qant()
        self.assertEqual(len(rows), 4)
        self.assertTrue(all(r["error"] is None for r in rows))
        self.assertTrue(all(v["exact_match_fraction"] == 1.0 for v in acts.values()))

    def test_simulator_f32acc_is_flat(self):
        x, w, exact = R.linear_inputs(11, 1024)
        self.assertLess(R.rel_mae(R.sim_linear(x, w, "f32acc"), exact), 0.005)


if __name__ == "__main__":
    unittest.main()
