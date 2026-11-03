"""CL-57: C training kernels against an independent NumPy SGD step (random data)."""
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

import experiments.cl53_mnist_hidden_sparsity.run as cl53
from experiments.cl57_sparse_input_training_cost.run import LR, CNet, build


def numpy_sgd(net, X, y, order):
    W1, b1 = net.W1.astype(np.float64), net.b1.astype(np.float64)
    W2, b2 = net.W2.astype(np.float64), net.b2.astype(np.float64)
    for i in order:
        x = X[i].astype(np.float64)
        pre = W1 @ x + b1
        h = np.maximum(pre, 0)
        z = W2 @ h + b2
        p = np.exp(z - z.max())
        p /= p.sum()
        p[y[i]] -= 1
        dh = (W2.T @ p) * (pre > 0)
        W2 -= LR * np.outer(p, h)
        b2 -= LR * p
        W1 -= LR * np.outer(dh, x)
        b1 -= LR * dh
    return W1, b1, W2, b2


@unittest.skipUnless(shutil.which("gcc"), "gcc not available")
class CL57Kernels(unittest.TestCase):
    def test_kernels_match_numpy_sgd_and_each_other(self):
        rng = np.random.default_rng(0)
        X = (rng.random((150, cl53.D)) * (rng.random((150, cl53.D)) < 0.2)).astype(np.float32)
        y = rng.integers(0, cl53.C, 150).astype(np.int64)
        order = rng.permutation(150).astype(np.int64)
        ref = numpy_sgd(cl53.Net(5, 32), X, y, order)
        with tempfile.TemporaryDirectory() as b:
            lib, _ = build(Path(b))
            nets = {}
            for sparse in (False, True):
                net = CNet(lib, 5, 32)
                net.train_epoch(X, y, order, sparse)
                nets[sparse] = net
                W1T, b1, W2, b2 = net.params()
                for got, want in ((W1T.T, ref[0]), (b1, ref[1]), (W2, ref[2]), (b2, ref[3])):
                    np.testing.assert_allclose(got, want, rtol=1e-3, atol=1e-4)
            for a, b_ in zip(nets[False].params(), nets[True].params()):
                self.assertTrue(np.array_equal(a, b_))


if __name__ == "__main__":
    unittest.main()
