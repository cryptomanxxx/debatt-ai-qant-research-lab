"""CL-74: the standalone pilot kit reproduces the CL-69/72/73 models exactly, on both float32 and the Q.ANT model."""
import unittest

import numpy as np

import experiments.cl69_qant_model_zoo.run as cl69
import experiments.cl72_application_mapping.run as cl72
import experiments.cl73_search_embeddings.run as cl73
from experiments.cl74_pilot_kit import kit


class CL74(unittest.TestCase):
    def setUp(self):
        self.X = np.random.default_rng(0).random((5, 784)).astype(np.float32)

    def test_mlp_matches_cl69(self):
        m = cl69.Net(3, *cl69.ZOO["mlp_h256x2"])
        W = {f"mlp.W{i}": L.p["W"].T.copy() for i, L in enumerate(m.layers)} | {f"mlp.b{i}": L.p["b"] for i, L in enumerate(m.layers)}
        np.testing.assert_allclose(kit.mlp(kit.Float32(), W, self.X), m.logits(self.X), rtol=1e-5, atol=1e-5)

    def test_cnn_matches_cl72_on_both_paths(self):
        m = cl72.CNN(3)
        W = {f"cnn.{k}": (v.T.copy() if k.endswith("_W") else v) for k, v in m.p.items()}
        np.testing.assert_allclose(kit.cnn(kit.Float32(), W, self.X), m.forward(self.X)[0], rtol=1e-5, atol=1e-5)
        sim = kit.cnn(kit.QantSim(), kit.prepared(kit.QantSim(), W, "cnn"), self.X)
        np.testing.assert_allclose(sim, m.qant_logits(self.X), rtol=1e-2, atol=1e-2)

    def test_encoder_matches_cl73_on_both_paths(self):
        p = cl73.init_params(3)
        W = {f"enc.{k}": v for k, v in p.items()}
        ids = cl73.pad([[cl73.CLS, 5, 9, 11], [cl73.CLS] + list(range(3, 40))])
        np.testing.assert_allclose(kit.encoder(kit.Float32(), W, ids), cl73.forward(p, ids), atol=1e-5)
        sim = kit.encoder(kit.QantSim(), kit.prepared(kit.QantSim(), W, "encoder"), ids)
        np.testing.assert_allclose(sim, cl73.forward(p, ids, cl73.Path_("qant")), atol=2e-3)

    def test_mac_counts(self):
        self.assertEqual(kit.macs("mlp"), 268_800)
        self.assertEqual(kit.macs("cnn"), cl72.lenet().summary()["total_macs"])
        self.assertEqual(kit.macs("encoder"), 2 * (128 * 49_152 + 2 * 128 * 128 * 64) + 64 * 64)

    def test_timed_wrapper_counts_matrix_time(self):
        m = cl69.Net(3, *cl69.ZOO["mlp_h256x2"])
        W = {f"mlp.W{i}": L.p["W"].T.copy() for i, L in enumerate(m.layers)} | {f"mlp.b{i}": L.p["b"] for i, L in enumerate(m.layers)}
        t = kit.Timed(kit.Float32())
        kit.mlp(t, W, self.X)
        self.assertGreater(t.t, 0)

    def test_thread_argument_both_forms(self):
        self.assertEqual(kit._threads_from_argv(["kit.py", "bench", "--threads", "4"]), "4")
        self.assertEqual(kit._threads_from_argv(["kit.py", "bench", "--threads=4"]), "4")
        self.assertEqual(kit._threads_from_argv(["kit.py", "bench"]), "1")


if __name__ == "__main__":
    unittest.main()
