"""CL-54: do counted savings become wall-time savings with compiled kernels? (Tier 1)

Retrains the CL-53 models for one seed (MNIST, downloaded if missing), then
times single-threaded C inference kernels (kernels.c, compiled at run time
with gcc) against the NumPy reference. See README.md and PREDICTIONS.md.

Run from the repository root:
    python -m experiments.cl54_compiled_sparse_kernels.run [--data-dir DIR]
"""
from __future__ import annotations

import experiments.cl53_mnist_hidden_sparsity.run as cl53  # sets single-threaded BLAS first

import argparse
import ctypes
import json
import platform
import statistics
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SEED = 11
CFLAGS = ["-O3", "-march=native", "-ffast-math", "-shared", "-fPIC"]
REPEATS = 7
MODELS = {
    "dense_H128": cl53.ARMS["dense_H128"],
    "dense_H256": cl53.ARMS["dense_H256"],
    "kwta_exact_H1024_k32": cl53.ARMS["kwta_exact_H1024_k32"],
    "kwta_exact_H1024_k64": cl53.ARMS["kwta_exact_H1024_k64"],
    "kwta_lowrank_H1024_k32": cl53.ARMS["kwta_lowrank_H1024_k32"],
    "kwta_lowrank_H1024_k64": cl53.ARMS["kwta_lowrank_H1024_k64"],
}


def train_like_cl53(cfg, seed, x_tr, y_tr):
    """Exactly the CL-53 training loop for one seed."""
    net = cl53.Net(seed, **cfg)
    rng = np.random.default_rng(seed + 1)
    batches = 0
    for _ in range(cl53.EPOCHS):
        order = rng.permutation(len(x_tr))
        for s in range(0, len(order), cl53.BATCH):
            idx = order[s:s + cl53.BATCH]
            net.train_batch(x_tr[idx], y_tr[idx])
            batches += 1
            if net.sel == "lowrank" and batches % cl53.REFRESH_BATCHES == 0:
                net.refresh()
    return net


def build_kernels(build_dir: Path):
    so = build_dir / "kernels.so"
    cmd = ["gcc", *CFLAGS, "-o", str(so), str(HERE / "kernels.c")]
    subprocess.run(cmd, check=True)
    lib = ctypes.CDLL(str(so))
    version = subprocess.run(["gcc", "--version"], capture_output=True, text=True).stdout.splitlines()[0]
    return lib, " ".join(cmd[:len(CFLAGS) + 1]), version


def f32(a):
    return np.ascontiguousarray(a, dtype=np.float32)


def ptr(a):
    return a.ctypes.data_as(ctypes.c_void_p)


class CModel:
    """Contiguous float32 copies of a trained cl53.Net plus kernel dispatch."""

    def __init__(self, lib, net, variant):
        self.lib, self.net, self.variant = lib, net, variant
        self.W1, self.W1T, self.b1 = f32(net.W1), f32(net.W1.T), f32(net.b1)
        self.W2, self.b2 = f32(net.W2), f32(net.b2)
        if net.sel == "lowrank":
            self.V, self.VT, self.A = f32(net.V), f32(net.V.T), f32(net.A)
        self.scratch = np.zeros(2 * net.H + cl53.D, np.float32)
        self.iscratch = np.zeros(net.H + cl53.D, np.int32)

    def predict(self, X):
        n, net, lib = len(X), self.net, self.lib
        out = np.zeros(n, np.int32)
        X = f32(X)
        common = (ptr(self.scratch), ptr(self.iscratch), ptr(out))
        I = ctypes.c_int
        if net.k is None:
            if self.variant == "sparse_input":
                lib.dense_sparse_input(ptr(X), I(n), I(cl53.D), I(net.H), I(cl53.C), ptr(self.W1T),
                                       ptr(self.b1), ptr(self.W2), ptr(self.b2), *common)
            else:
                lib.dense(ptr(X), I(n), I(cl53.D), I(net.H), I(cl53.C), ptr(self.W1),
                          ptr(self.b1), ptr(self.W2), ptr(self.b2), *common)
        elif net.sel == "exact":
            lib.kwta_exact(ptr(X), I(n), I(cl53.D), I(net.H), I(cl53.C), I(net.k),
                           I(self.variant == "sparse_input"), ptr(self.W1), ptr(self.W1T), ptr(self.b1),
                           ptr(self.W2), ptr(self.b2), *common)
        else:
            lib.kwta_lowrank(ptr(X), I(n), I(cl53.D), I(net.H), I(cl53.C), I(net.k), I(net.r), I(net.c),
                             I(self.variant == "sparse_input"), ptr(self.W1), ptr(self.V), ptr(self.VT),
                             ptr(self.A), ptr(self.b1), ptr(self.W2), ptr(self.b2), *common)
        return out.astype(np.int64)


def counted_terms(net, variant, nnz):
    """Analytic per-sample work for the kernel actually run (fan-in nnz if sparse input)."""
    D, C, H = cl53.D, cl53.C, net.H
    fan = nnz if variant == "sparse_input" else D
    if net.k is None:
        return fan * H + C * H
    if net.sel == "exact":
        return fan * H + H + C * net.k
    return fan * net.r + H * net.r + H + fan * net.c + net.c + C * net.k


def median_time(fn):
    times = []
    for _ in range(REPEATS):
        t = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t)
    return {"median_s": statistics.median(times), "min_s": min(times), "max_s": max(times)}


def run(data_dir: Path, build_dir: Path, models=MODELS, timing=True, n_test=None):
    x_tr, y_tr, x_te, y_te = cl53.load_mnist(data_dir)
    if n_test:
        x_te, y_te = x_te[:n_test], y_te[:n_test]
    lib, flags, gcc = build_kernels(build_dir)
    nnz = float((x_te != 0).sum(1).mean())
    cl53_results = json.loads((HERE.parent / "cl53_mnist_hidden_sparsity" / "results.json").read_text())
    out = {"experiment": "CL-54", "seed": SEED, "cflags": flags, "compiler": gcc,
           "cpu": platform.processor() or platform.machine(), "numpy": np.__version__,
           "mean_nonzero_pixels_per_test_sample": nnz, "kernels": {}}
    timing_out = {}
    for name, cfg in models.items():
        net = train_like_cl53(cfg, SEED, x_tr, y_tr)
        ref_pred = net.predict(x_te)
        acc = float((ref_pred == y_te).mean())
        seed_idx = cl53_results["seeds"].index(SEED)
        out.setdefault("retrained_accuracy_matches_cl53", {})[name] = (
            acc == cl53_results["arms"][name]["test_accuracy_per_seed"][seed_idx])
        if timing:
            timing_out[f"numpy_{name}"] = median_time(lambda: net.predict_sparse(x_te))
            if name == "dense_H256":
                timing_out["numpy_dense_H256_AA"] = median_time(lambda: net.predict_sparse(x_te))
        for variant in ("full_input", "sparse_input"):
            cm = CModel(lib, net, variant)
            pred = cm.predict(x_te)
            key = f"c_{name}_{variant}"
            out["kernels"][key] = {
                "model": name, "variant": variant,
                "agreement_with_numpy": float((pred == ref_pred).mean()),
                "test_accuracy": float((pred == y_te).mean()),
                "counted_terms_per_sample": counted_terms(net, variant, nnz),
            }
            if timing:
                timing_out[key] = median_time(lambda: cm.predict(x_te))
                if key == "c_dense_H256_full_input":
                    timing_out["c_dense_H256_full_input_AA"] = median_time(lambda: cm.predict(x_te))
    return out, timing_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=HERE.parent / "cl53_mnist_hidden_sparsity" / "data")
    args = ap.parse_args()
    with tempfile.TemporaryDirectory() as build:
        results, timing = run(args.data_dir, Path(build))
    (HERE / "results.json").write_text(json.dumps(results, indent=1, sort_keys=True) + "\n")
    (HERE / "timing.json").write_text(json.dumps(timing, indent=1, sort_keys=True) + "\n")
    ref_t = timing["c_dense_H256_full_input"]["median_s"]
    ref_w = results["kernels"]["c_dense_H256_full_input"]["counted_terms_per_sample"]
    print("retrained accuracy matches CL-53:", results["retrained_accuracy_matches_cl53"])
    for key, k in results["kernels"].items():
        t = timing[key]["median_s"]
        print(f"{key:44s} agree {k['agreement_with_numpy']:.4f} acc {100 * k['test_accuracy']:.2f}% "
              f"work {k['counted_terms_per_sample'] / ref_w:5.2f} time {t * 1000:7.1f} ms ({t / ref_t:5.2f})")
    for key, v in timing.items():
        if key.startswith("numpy_") or key.endswith("_AA"):
            print(f"{key:44s} time {v['median_s'] * 1000:7.1f} ms ({v['median_s'] / ref_t:5.2f})")


if __name__ == "__main__":
    main()
