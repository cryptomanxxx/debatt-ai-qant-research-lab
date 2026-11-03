"""CL-54: compiled kernels agree with the NumPy reference (random data, no timing)."""
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

import experiments.cl53_mnist_hidden_sparsity.run as cl53
from experiments.cl54_compiled_sparse_kernels.run import CModel, build_kernels


@unittest.skipUnless(shutil.which("gcc"), "gcc not available")
class CL54KernelAgreement(unittest.TestCase):
    def test_all_kernels_match_numpy_predictions(self):
        rng = np.random.default_rng(0)
        # Sparse non-negative inputs, like MNIST pixels.
        X = (rng.random((200, cl53.D)) * (rng.random((200, cl53.D)) < 0.2)).astype(np.float32)
        y = rng.integers(0, cl53.C, 200)
        configs = (dict(H=64), dict(H=256, k=16, sel="exact"),
                   dict(H=256, k=16, sel="lowrank", r=16, c=64))
        with tempfile.TemporaryDirectory() as build:
            lib, _, _ = build_kernels(Path(build))
            for cfg in configs:
                net = cl53.Net(3, **cfg)
                for s in range(0, 200, 50):
                    net.train_batch(X[s:s + 50], y[s:s + 50])
                ref = net.predict(X)
                for variant in ("full_input", "sparse_input"):
                    pred = CModel(lib, net, variant).predict(X)
                    self.assertGreaterEqual((pred == ref).mean(), 0.99, (cfg, variant))


if __name__ == "__main__":
    unittest.main()
