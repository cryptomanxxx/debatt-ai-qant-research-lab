"""CL-58: quantization-aware training for the Q.ANT Fourier/KAN layer.

Part 1 (local, Tier 0/1): train in NumPy, evaluate float32 and simulated bfloat16.
Part 2 (--qant, Tier 2, GitHub Actions in the Q.ANT image): also evaluate on the
Q.ANT CPU backend exactly as Exp015 (calc_kan_layer_fprop + add_bias_fprop).
See README.md and PREDICTIONS.md.

Run from the repository root:
    python -m experiments.cl58_qant_quantization_aware_training.run [--qant] [--data-dir DIR]
"""
from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_var] = "1"

import argparse
import gzip
import hashlib
import json
import multiprocessing as mp
import platform
import statistics
import time
import urllib.request
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
URL = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"
FILES = {  # SHA-256 pinned; MD5 matches torchvision's official Fashion-MNIST checksums
    "train-images-idx3-ubyte.gz": "3aede38d61863908ad78613f6a32ed271626dd12800ba2636569512369268a84",
    "train-labels-idx1-ubyte.gz": "a04f17134ac03560a47e3764e11b92fc97de4d1bfaf8ba1a3aa29af54cc90845",
    "t10k-images-idx3-ubyte.gz": "346e55b948d973a97e58d2351dde16a484bd415d4595297633bb08f03db6a073",
    "t10k-labels-idx1-ubyte.gz": "67da17c76eaffca5446c3361aaab5c3cd6d1c2608764d35dfb1850b086bf8dd5",
}
SEEDS = (11, 22, 33)
GRIDS = (4, 8)
ARMS = ("exact_cosine", "bf16_ste", "bf16_ste_noise")
EPOCHS, BATCH, LR = 3, 128, 1e-3
NOISE_SIGMA = 0.003
D, C = 784, 10


# ------------------------------------------------------------------ data

def load_fashion_mnist(data_dir: Path):
    data_dir.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for name, digest in FILES.items():
        path = data_dir / name
        if not path.exists():
            urllib.request.urlretrieve(URL + name, path)
        raw_gz = path.read_bytes()
        if hashlib.sha256(raw_gz).hexdigest() != digest:
            raise RuntimeError(f"{name}: SHA-256 mismatch")
        raw = gzip.decompress(raw_gz)
        magic = int.from_bytes(raw[:4], "big")
        if magic == 2051:
            n, r, c = (int.from_bytes(raw[4 + 4 * i:8 + 4 * i], "big") for i in range(3))
            arrays[name] = np.frombuffer(raw, np.uint8, offset=16).reshape(n, r * c)
        elif magic == 2049:
            arrays[name] = np.frombuffer(raw, np.uint8, offset=8)
        else:
            raise RuntimeError(f"{name}: bad IDX magic {magic}")
    return (arrays["train-images-idx3-ubyte.gz"].astype(np.float32) / 255,
            arrays["train-labels-idx1-ubyte.gz"].astype(np.int64),
            arrays["t10k-images-idx3-ubyte.gz"].astype(np.float32) / 255,
            arrays["t10k-labels-idx1-ubyte.gz"].astype(np.int64))


# ------------------------------------------------------------------ bfloat16

def bf16(a):
    """Round float32 to the nearest bfloat16 (ties to even), returned as float32."""
    a = np.ascontiguousarray(a, dtype=np.float32)
    u = a.view(np.uint32)
    rounded = (u + (((u >> 16) & 1) + np.uint32(0x7FFF))) & np.uint32(0xFFFF0000)
    return rounded.view(np.float32)


# ------------------------------------------------------------------ model

class QFourier:
    """logits[c] = b[c] + sum_{i,k} A[c,i,k] cos(k x_i + phi[c,i,k]) (Exp015 layer)."""

    def __init__(self, seed, grid):
        rng = np.random.default_rng(seed)
        std = (2 / ((D + C) * grid)) ** 0.5
        self.A = (rng.standard_normal((C, D, grid)) * std).astype(np.float32)
        self.phi = rng.uniform(-np.pi / 2, np.pi / 2, (C, D, grid)).astype(np.float32)
        self.b = np.zeros(C, np.float32)
        self.k = np.arange(1, grid + 1, dtype=np.float32)
        self.m = {n: np.zeros_like(getattr(self, n)) for n in ("A", "phi", "b")}
        self.v = {n: np.zeros_like(getattr(self, n)) for n in ("A", "phi", "b")}
        self.t = 0

    def _act(self, X, quant, noise_rng=None):
        x, A, phi = (bf16(X), bf16(self.A), bf16(self.phi)) if quant else (X, self.A, self.phi)
        z = x[:, None, :, None] * self.k[None, None, None, :] + phi[None]
        act = np.cos(z)
        if quant:
            act = bf16(act)
        if noise_rng is not None:
            act = act + noise_rng.normal(0, NOISE_SIGMA, act.shape).astype(np.float32)
        return z, act, A

    def logits(self, X, quant=False):
        _, act, A = self._act(X, quant)
        b = bf16(self.b) if quant else self.b
        return np.einsum("bcik,cik->bc", act, A) + b

    def train_batch(self, X, y, arm, noise_rng):
        quant = arm != "exact_cosine"
        z, act, A = self._act(X, quant, noise_rng if arm == "bf16_ste_noise" else None)
        b = bf16(self.b) if quant else self.b
        logits = np.einsum("bcik,cik->bc", act, A) + b
        logits -= logits.max(1, keepdims=True)
        p = np.exp(logits)
        p /= p.sum(1, keepdims=True)
        p[np.arange(len(y)), y] -= 1
        G = (p / len(y)).astype(np.float32)
        # Straight-through: gradients use the (quantized) forward values, applied to float32 weights.
        grads = {"A": np.einsum("bc,bcik->cik", G, act),
                 "phi": -A * np.einsum("bc,bcik->cik", G, np.sin(z)),
                 "b": G.sum(0)}
        self.t += 1
        for name, g in grads.items():
            self.m[name] = 0.9 * self.m[name] + 0.1 * g
            self.v[name] = 0.999 * self.v[name] + 0.001 * g * g
            mh = self.m[name] / (1 - 0.9 ** self.t)
            vh = self.v[name] / (1 - 0.999 ** self.t)
            param = getattr(self, name)
            param -= (LR * mh / (np.sqrt(vh) + 1e-8)).astype(np.float32)

    def predict(self, X, quant=False, batch=250):
        return np.concatenate([self.logits(X[s:s + batch], quant).argmax(1) for s in range(0, len(X), batch)])


def qant_predict(model, X, batch=250):
    """Q.ANT CPU backend inference, identical in form to Exp015's qant_pred."""
    import qant_native_computing_toolkit.ai as q_ai
    from ml_dtypes import bfloat16
    out = []
    for s in range(0, len(X), batch):
        o = q_ai.calc_kan_layer_fprop(X[s:s + batch].astype(bfloat16), model.phi.astype(bfloat16),
                                      model.A.astype(bfloat16), model.k.astype(bfloat16))
        o = q_ai.add_bias_fprop(o, model.b.astype(bfloat16))
        out.append(np.asarray(o, dtype=np.float32).argmax(1))
    return np.concatenate(out)


# ------------------------------------------------------------------ run

def run_one(args):
    seed, grid, arm, data_dir, qant, n_train = args
    x_tr, y_tr, x_te, y_te = load_fashion_mnist(Path(data_dir))
    if n_train:
        x_tr, y_tr = x_tr[:n_train], y_tr[:n_train]
    model = QFourier(seed, grid)
    rng = np.random.default_rng(seed + 1)
    noise_rng = np.random.default_rng(seed + 2)
    start = time.perf_counter()
    for _ in range(EPOCHS):
        order = rng.permutation(len(x_tr))
        for s in range(0, len(order), BATCH):
            idx = order[s:s + BATCH]
            model.train_batch(x_tr[idx], y_tr[idx], arm, noise_rng)
    train_s = time.perf_counter() - start
    ref = model.predict(x_te)
    sim = model.predict(x_te, quant=True)
    row = {"seed": seed, "grid": grid, "arm": arm, "train_seconds": train_s,
           "reference_accuracy": float((ref == y_te).mean()),
           "simulated_bf16_accuracy": float((sim == y_te).mean()),
           "disagreements_ref_vs_sim": int((ref != sim).sum())}
    if qant:
        q = qant_predict(model, x_te)
        row.update({"qant_accuracy": float((q == y_te).mean()),
                    "disagreements_ref_vs_qant": int((ref != q).sum()),
                    "disagreements_sim_vs_qant": int((sim != q).sum())})
    return row


def summarize(rows):
    out = {}
    for grid in sorted({r["grid"] for r in rows}):
        for arm in ARMS:
            rs = [r for r in rows if r["grid"] == grid and r["arm"] == arm]
            if not rs:
                continue
            keys = [k for k in rs[0] if k not in ("seed", "grid", "arm")]
            out[f"g{grid}_{arm}"] = {k: {"mean": statistics.fmean(r[k] for r in rs),
                                         "per_seed": [r[k] for r in rs]} for k in keys}
    return out


def run(data_dir: Path, qant=False, seeds=SEEDS, grids=GRIDS, arms=ARMS, n_train=None, processes=3):
    load_fashion_mnist(data_dir)  # download/verify once before forking
    jobs = [(s, g, a, str(data_dir), qant, n_train) for g in grids for a in arms for s in seeds]
    if processes > 1:
        with mp.get_context("fork").Pool(processes) as pool:
            rows = pool.map(run_one, jobs)
    else:
        rows = list(map(run_one, jobs))
    meta = {"experiment": "CL-58", "dataset": "FashionMNIST", "sha256": FILES, "seeds": list(seeds),
            "grids": list(grids), "epochs": EPOCHS, "batch": BATCH, "lr": LR, "noise_sigma": NOISE_SIGMA,
            "backend": "qant-cpu + numpy" if qant else "numpy (bf16 simulated)",
            "python": platform.python_version(), "numpy": np.__version__}
    return {**meta, "rows": rows, "summary": summarize(rows)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qant", action="store_true", help="also evaluate on the Q.ANT CPU backend")
    ap.add_argument("--data-dir", type=Path, default=HERE / "data")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--processes", type=int, default=3)
    args = ap.parse_args()
    results = run(args.data_dir, qant=args.qant, processes=args.processes)
    out = args.out or HERE / ("results_qant.json" if args.qant else "results_local.json")
    out.write_text(json.dumps(results, indent=1, sort_keys=True) + "\n")
    for key, s in results["summary"].items():
        line = (f"{key:22s} ref {s['reference_accuracy']['mean']:.4f} sim {s['simulated_bf16_accuracy']['mean']:.4f} "
                f"dis(ref,sim) {s['disagreements_ref_vs_sim']['mean']:6.1f}")
        if "qant_accuracy" in s:
            line += (f" qant {s['qant_accuracy']['mean']:.4f} dis(ref,qant) {s['disagreements_ref_vs_qant']['mean']:6.1f}"
                     f" dis(sim,qant) {s['disagreements_sim_vs_qant']['mean']:6.1f}")
        print(line)


if __name__ == "__main__":
    main()
