"""CL-67: training the Q.ANT Fourier/KAN layer through the simulated accumulation (Tier 0).

Arms: exact (CL-58 exact_cosine), seq_ste (every training forward pass through the
CL-62 sequential-bfloat16 simulator, straight-through gradients), finetune (2 exact
epochs + 1 seq_ste epoch). The sequential sum runs in a small C kernel that is
bit-identical to the NumPy simulator. See README.md and PREDICTIONS.md.

Run from the repository root:
    python -m experiments.cl67_accumulation_aware_training.run [--processes N]
"""
from __future__ import annotations

import experiments.cl62_qant_accumulation_simulator.run as cl62  # sets single-threaded BLAS first

import argparse
import ctypes
import json
import multiprocessing as mp
import statistics
import subprocess
import tempfile
from pathlib import Path

import numpy as np

cl58 = cl62.cl58
bf16 = cl58.bf16
HERE = Path(__file__).resolve().parent
SEEDS = cl58.SEEDS
GRID = 4
ARMS = ("exact", "seq_ste", "finetune")
CFLAGS = ["-O2", "-shared", "-fPIC", "-fno-fast-math", "-ffp-contract=off"]
_LIB = None


def lib():
    global _LIB
    if _LIB is None:
        so = Path(tempfile.mkdtemp(prefix="cl67_")) / "seq_bf16.so"
        subprocess.run(["gcc", *CFLAGS, "-o", str(so), str(HERE / "seq_bf16.c")], check=True)
        _LIB = ctypes.CDLL(str(so))
        _LIB.seq_bf16.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_long, ctypes.c_long]
    return _LIB


def seq_sum(t):
    """t (B, C, n) float32 -> (B, C), sequential bf16 sum (C kernel)."""
    t = np.ascontiguousarray(t, dtype=np.float32)
    out = np.empty(t.shape[:2], np.float32)
    lib().seq_bf16(t.ctypes.data, out.ctypes.data, t.shape[0] * t.shape[1], t.shape[2])
    return out


def sim_logits(model, X):
    """CL-62 'seq' variant (one Q.ANT call per logit), using the C kernel."""
    _, act, A = model._act(X, quant=True)
    t = bf16(act * A[None]).reshape(len(X), cl58.C, -1)
    return bf16(bf16(seq_sum(t)) + bf16(model.b))


class AccModel(cl58.QFourier):
    def train_seq_ste(self, X, y):
        z, act, A = self._act(X, quant=True)
        t = bf16(act * A[None]).reshape(len(X), cl58.C, -1)
        logits = bf16(bf16(seq_sum(t)) + bf16(self.b))
        logits = logits - logits.max(1, keepdims=True)
        p = np.exp(logits)
        p /= p.sum(1, keepdims=True)
        p[np.arange(len(y)), y] -= 1
        G = (p / len(y)).astype(np.float32)
        # Straight-through: float32 gradient formulas at the bf16 operands and cosines.
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
            param -= (cl58.LR * mh / (np.sqrt(vh) + 1e-8)).astype(np.float32)


def train(seed, arm, x_tr, y_tr):
    model = AccModel(seed, GRID)
    rng = np.random.default_rng(seed + 1)
    noise_rng = np.random.default_rng(seed + 2)
    for epoch in range(cl58.EPOCHS):
        through_sim = arm == "seq_ste" or (arm == "finetune" and epoch == cl58.EPOCHS - 1)
        order = rng.permutation(len(x_tr))
        for s in range(0, len(order), cl58.BATCH):
            idx = order[s:s + cl58.BATCH]
            if through_sim:
                model.train_seq_ste(x_tr[idx], y_tr[idx])
            else:
                model.train_batch(x_tr[idx], y_tr[idx], "exact_cosine", noise_rng)
    return model


def job(args):
    seed, arm, data_dir = args
    x_tr, y_tr, x_te, y_te = cl58.load_fashion_mnist(Path(data_dir))
    model = train(seed, arm, x_tr, y_tr)
    ref = model.predict(x_te)
    sim = np.concatenate([sim_logits(model, x_te[s:s + 250]).argmax(1) for s in range(0, len(x_te), 250)])
    return {"arm": arm, "seed": seed, "reference_accuracy": float((ref == y_te).mean()),
            "qant_path_accuracy": float((sim == y_te).mean()), "disagreements": int((ref != sim).sum())}


def run(data_dir, processes):
    cl58.load_fashion_mnist(data_dir)
    lib()
    with mp.get_context("fork").Pool(processes) as pool:
        rows = pool.map(job, [(s, a, str(data_dir)) for a in ARMS for s in SEEDS])
    summary = {a: {k: statistics.fmean(r[k] for r in rows if r["arm"] == a)
                   for k in ("reference_accuracy", "qant_path_accuracy", "disagreements")} for a in ARMS}
    return {"experiment": "CL-67", "dataset": "FashionMNIST", "grid": GRID, "seeds": list(SEEDS),
            "epochs": cl58.EPOCHS, "cflags": CFLAGS, "numpy": np.__version__,
            "evaluation": "CL-62 simulator, one Q.ANT call per logit (bf16 operands/cos/products, sequential bf16 sum, bf16 bias)",
            "rows": rows, "summary": summary}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=cl58.HERE / "data")
    ap.add_argument("--processes", type=int, default=3)
    args = ap.parse_args()
    res = run(args.data_dir, args.processes)
    (HERE / "results.json").write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")
    for r in res["rows"]:
        print(r)
    for a, s in res["summary"].items():
        print(f"{a:9s} ref {s['reference_accuracy']:.4f} qant-path {s['qant_path_accuracy']:.4f} dis {s['disagreements']:.1f}")


if __name__ == "__main__":
    main()
