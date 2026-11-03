"""CL-71: the Q.ANT evaluation path runs end to end on a fake backend and equals the simulator path there."""
import sys
import types
import unittest
from unittest import mock

import numpy as np

import experiments.cl71_qant_zoo_confirmation.run as R


def fake_modules():
    bf16 = R.cl58.bf16
    ai = types.ModuleType("qant_native_computing_toolkit.ai")
    f = lambda v: bf16(np.asarray(v, np.float32))  # the fake dtype is float32, so round inputs to bf16 here
    ai.linear_fprop = lambda x, w: bf16(f(x) @ f(w).T)
    ai.calc_kan_layer_fprop = lambda x, p, a, k: bf16(R.cl69.cl67.seq_sum(
        bf16(bf16(np.cos(f(x)[:, None, :, None] * f(k) + f(p)[None])) * f(a)[None]).reshape(len(f(x)), f(a).shape[0], -1)))
    ai.relu_fprop = lambda v: np.maximum(f(v), 0)
    ai.sigmoid_fprop = lambda v: bf16(1 / (1 + np.exp(-f(v))))
    ai.add_bias_fprop = lambda a, b: bf16(f(a) + f(b))
    pkg = types.ModuleType("qant_native_computing_toolkit")
    pkg.ai = ai
    md = types.ModuleType("ml_dtypes")
    md.bfloat16 = np.float32
    return {"qant_native_computing_toolkit": pkg, "qant_native_computing_toolkit.ai": ai, "ml_dtypes": md}


class CL71(unittest.TestCase):
    def test_fake_qant_path_equals_simulator_path(self):
        X = np.random.default_rng(0).random((6, 784)).astype(np.float32)
        with mock.patch.dict(sys.modules, fake_modules()):
            be = R.Qant()
            for name in ("mlp_sig_h256", "lin128_kanhead", "kan32_linhead"):
                spec, act = R.cl69.ZOO[name]
                model = R.cl69.Net(1, spec, act)
                np.testing.assert_array_equal(be.logits(model, X), model.qant_logits(X), err_msg=name)
            model = R.cl69.Net(2, *R.cl69.ZOO["kan_g4"])
            np.testing.assert_array_equal(be.logits(model, X, 16), model.qant_logits(X, 16))

    def test_pack_round_trip(self):
        preds = {"a_s11_ref": np.arange(5, dtype=np.uint8), "a_s11_qant": np.ones(5, np.uint8)}
        keys, data, digest = R.pack(preds)
        out = R.unpack({"keys": keys, "data": data, "sha256_raw": digest})
        for k in preds:
            np.testing.assert_array_equal(out[k], preds[k])


if __name__ == "__main__":
    unittest.main()
