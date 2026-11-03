"""CL-57: training-step cost with sparse inputs (Tier 1, MNIST reused).

Per-sample SGD in C (train_kernels.c, compiled at run time) with and without
skipping zero pixels, plus a NumPy batch-64 dense reference.
See README.md and PREDICTIONS.md.

Run from the repository root:
    python -m experiments.cl57_sparse_input_training_cost.run [--data-dir DIR]
"""
from __future__ import annotations

import experiments.cl53_mnist_hidden_sparsity.run as cl53  # sets single-threaded BLAS first

import argparse
import ctypes
import json
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SEEDS = (11, 22, 33)
HIDDEN = (128, 256)
LR = 0.01
NUMPY_BATCH, NUMPY_LR = 64, 0.05
CFLAGS = ["-O3", "-march=native", "-ffast-math", "-shared", "-fPIC"]
D, C = cl53.D, cl53.C


def build(build_dir: Path):
    so = build_dir / "train_kernels.so"
    subprocess.run(["gcc", *CFLAGS, "-o", str(so), str(HERE / "train_kernels.c"), "-lm"], check=True)
    version = subprocess.run(["gcc", "--version"], capture_output=True, text=True).stdout.splitlines()[0]
    return ctypes.CDLL(str(so)), version


def ptr(a):
    return a.ctypes.data_as(ctypes.c_void_p)


class CNet:
    """Parameters in kernel layout, initialised exactly as cl53.Net."""

    def __init__(self, lib, seed, H):
        ref = cl53.Net(seed, H)
        self.lib, self.H = lib, H
        self.W1T = np.ascontiguousarray(ref.W1.T, dtype=np.float32)
        self.b1, self.W2, self.b2 = ref.b1.copy(), ref.W2.copy(), ref.b2.copy()
        self.scratch = np.zeros(2 * H + C, np.float32)
        self.iscratch = np.zeros(D, np.int32)

    def train_epoch(self, X, y, order, sparse_input):
        I = ctypes.c_int
        self.lib.train_epoch(ptr(X), ptr(y), ptr(order), I(len(order)), I(D), I(self.H), I(C),
                             ptr(self.W1T), ptr(self.b1), ptr(self.W2), ptr(self.b2),
                             ctypes.c_float(LR), I(int(sparse_input)), ptr(self.scratch), ptr(self.iscratch))

    def predict(self, X):
        out = np.zeros(len(X), np.int64)
        I = ctypes.c_int
        self.lib.predict(ptr(X), I(len(X)), I(D), I(self.H), I(C), ptr(self.W1T), ptr(self.b1),
                         ptr(self.W2), ptr(self.b2), ptr(self.scratch), ptr(self.iscratch), ptr(out))
        return out

    def params(self):
        return (self.W1T, self.b1, self.W2, self.b2)


def numpy_batch_epoch(seed, H, x_tr, y_tr):
    net = cl53.Net(seed, H)
    order = np.random.default_rng(seed + 1).permutation(len(x_tr))
    old = cl53.LR
    cl53.LR = NUMPY_LR
    try:
        start = time.perf_counter()
        for s in range(0, len(order), NUMPY_BATCH):
            idx = order[s:s + NUMPY_BATCH]
            net.train_batch(x_tr[idx], y_tr[idx])
        seconds = time.perf_counter() - start
    finally:
        cl53.LR = old
    return net, seconds


def counted_terms_per_step(H, fan):
    """Forward + backward + update weight terms per training sample (biases excluded)."""
    return (fan * H + C * H) + C * H + C * H + fan * H


def run(data_dir: Path, build_dir: Path, seeds=SEEDS, hidden=HIDDEN, n_train=None, timing=True):
    x_tr, y_tr, x_te, y_te = cl53.load_mnist(data_dir)
    if n_train:
        x_tr, y_tr = x_tr[:n_train], y_tr[:n_train]
    x_tr = np.ascontiguousarray(x_tr, np.float32)
    y_tr = np.ascontiguousarray(y_tr, np.int64)
    x_te = np.ascontiguousarray(x_te, np.float32)
    lib, gcc = build(build_dir)
    nnz = float((x_tr != 0).sum(1).mean())
    out = {"experiment": "CL-57", "seeds": list(seeds), "lr": LR, "cflags": " ".join(CFLAGS[:3]),
           "compiler": gcc, "numpy": np.__version__, "mean_nonzero_pixels_per_train_sample": nnz,
           "runs": {}}
    timing_out = {}
    for H in hidden:
        for seed in seeds:
            order = np.random.default_rng(seed + 1).permutation(len(x_tr)).astype(np.int64)
            res = {}
            nets = {}
            for variant, sparse in (("dense", False), ("sparse_input", True)):
                net = CNet(lib, seed, H)
                start = time.perf_counter()
                net.train_epoch(x_tr, y_tr, order, sparse)
                seconds = time.perf_counter() - start
                nets[variant] = net
                res[variant] = {"test_accuracy": float((net.predict(x_te) == y_te).mean()),
                                "counted_terms_per_step": counted_terms_per_step(H, nnz if sparse else D)}
                if timing:
                    timing_out[f"c_{variant}_H{H}_seed{seed}"] = seconds
            res["weights_bit_identical"] = all(
                np.array_equal(a, b) for a, b in zip(nets["dense"].params(), nets["sparse_input"].params()))
            if timing and H == 256 and seed == seeds[0]:
                aa = CNet(lib, seed, H)
                start = time.perf_counter()
                aa.train_epoch(x_tr, y_tr, order, False)
                timing_out[f"c_dense_H{H}_seed{seed}_AA"] = time.perf_counter() - start
            if H == 256:
                net, seconds = numpy_batch_epoch(seed, H, x_tr, y_tr)
                res["numpy_batch64_test_accuracy"] = float((net.predict(x_te) == y_te).mean())
                if timing:
                    timing_out[f"numpy_batch64_dense_H{H}_seed{seed}"] = seconds
            out["runs"][f"H{H}_seed{seed}"] = res
    return out, timing_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=HERE.parent / "cl53_mnist_hidden_sparsity" / "data")
    args = ap.parse_args()
    with tempfile.TemporaryDirectory() as build:
        results, timing = run(args.data_dir, Path(build))
    (HERE / "results.json").write_text(json.dumps(results, indent=1, sort_keys=True) + "\n")
    (HERE / "timing.json").write_text(json.dumps(timing, indent=1, sort_keys=True) + "\n")
    print("mean nonzero pixels:", round(results["mean_nonzero_pixels_per_train_sample"], 1))
    for key, r in results["runs"].items():
        H, seed = key.split("_")
        td, ts = timing[f"c_dense_{key}"], timing[f"c_sparse_input_{key}"]
        line = (f"{key:12s} identical {r['weights_bit_identical']} acc dense {r['dense']['test_accuracy']:.4f} "
                f"sparse {r['sparse_input']['test_accuracy']:.4f} | dense {td:6.2f}s sparse {ts:6.2f}s "
                f"ratio {ts / td:.2f} work ratio "
                f"{r['sparse_input']['counted_terms_per_step'] / r['dense']['counted_terms_per_step']:.2f}")
        if "numpy_batch64_test_accuracy" in r:
            tn = timing[f"numpy_batch64_dense_{key}"]
            line += f" | numpy b64 {tn:5.2f}s acc {r['numpy_batch64_test_accuracy']:.4f} sparse/numpy {ts / tn:.2f}"
        print(line)
    print("A/A:", {k: round(v, 2) for k, v in timing.items() if "seed11" in k and "dense_H256" in k and "c_" in k})


if __name__ == "__main__":
    main()
