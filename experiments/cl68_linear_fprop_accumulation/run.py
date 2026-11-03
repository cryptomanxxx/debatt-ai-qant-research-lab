"""CL-68: does Q.ANT's linear_fprop accumulate like calc_kan_layer_fprop? (Tier 0, phase A)

NumPy reimplementation of the archived Exp008 (MNIST) and Exp009 (Fashion-MNIST) ReLU
MLPs, evaluated along the archived Q.ANT path (linear_fprop -> add_bias_fprop ->
relu_fprop ..., softmax_fprop -> argmax) with two linear_fprop variants: float32
accumulation (f32acc) or sequential bfloat16 accumulation (seq, C kernel). Simulated
disagreement counts are compared with the archived Q.ANT counts. See PREDICTIONS.md.

Run from the repository root:
    python -m experiments.cl68_linear_fprop_accumulation.run [--processes N]
"""
from __future__ import annotations

import experiments.cl58_qant_quantization_aware_training.run as cl58  # sets single-threaded BLAS first
import experiments.cl53_mnist_hidden_sparsity.run as cl53

import argparse
import ctypes
import json
import multiprocessing as mp
import statistics
import subprocess
import tempfile
from pathlib import Path

import numpy as np

bf16 = cl58.bf16
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SEEDS = (11, 22, 33)
EPOCHS, BATCH, LR = 3, 128, 1e-3
SETS = {
    "MNIST": {"archive": "exp008_full_mnist_validation",
              "topologies": {"h32_512": [32, 512], "h64_512": [64, 512], "h128_512": [128, 512],
                             "h128": [128], "h256": [256], "h512": [512]}},
    "FashionMNIST": {"archive": "exp009_cross_dataset_validation",
                     "topologies": {"h64_512": [64, 512], "h128_512": [128, 512], "h128": [128],
                                    "h256": [256], "h512": [512]}},
}
CFLAGS = ["-O2", "-shared", "-fPIC", "-fno-fast-math", "-ffp-contract=off"]
_LIB = None


def lib():
    global _LIB
    if _LIB is None:
        so = Path(tempfile.mkdtemp(prefix="cl68_")) / "seq_dot.so"
        subprocess.run(["gcc", *CFLAGS, "-o", str(so), str(HERE / "seq_dot.c")], check=True)
        _LIB = ctypes.CDLL(str(so))
        _LIB.seq_dot.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_long] * 3
    return _LIB


def seq_dot(x, w):
    x = np.ascontiguousarray(x, np.float32)
    w = np.ascontiguousarray(w, np.float32)
    out = np.empty((x.shape[0], w.shape[0]), np.float32)
    lib().seq_dot(x.ctypes.data, w.ctypes.data, out.ctypes.data, x.shape[0], w.shape[0], x.shape[1])
    return out


def load(name, data_dir=None):
    if name == "MNIST":
        return cl53.load_mnist(Path(data_dir or ROOT / "experiments/cl53_mnist_hidden_sparsity/data"))
    return cl58.load_fashion_mnist(Path(data_dir or cl58.HERE / "data"))


class MLP:
    """784 -> widths -> 10 ReLU MLP with PyTorch nn.Linear default init and Adam."""

    def __init__(self, seed, widths):
        rng = np.random.default_rng(seed)
        dims = [784] + list(widths) + [10]
        self.W, self.b = [], []
        for i in range(len(dims) - 1):
            bound = 1 / np.sqrt(dims[i])
            self.W.append(rng.uniform(-bound, bound, (dims[i + 1], dims[i])).astype(np.float32))
            self.b.append(rng.uniform(-bound, bound, dims[i + 1]).astype(np.float32))
        self.params = self.W + self.b
        self.m = [np.zeros_like(p) for p in self.params]
        self.v = [np.zeros_like(p) for p in self.params]
        self.t = 0

    def forward(self, X):
        acts = [X]
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = acts[-1] @ W.T + b
            acts.append(np.maximum(z, 0) if i < len(self.W) - 1 else z)
        return acts

    def train_batch(self, X, y):
        acts = self.forward(X)
        z = acts[-1] - acts[-1].max(1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(1, keepdims=True)
        p[np.arange(len(y)), y] -= 1
        g = (p / len(y)).astype(np.float32)
        gW, gb = [None] * len(self.W), [None] * len(self.W)
        for i in range(len(self.W) - 1, -1, -1):
            gW[i] = g.T @ acts[i]
            gb[i] = g.sum(0)
            if i:
                g = (g @ self.W[i]) * (acts[i] > 0)
        self.t += 1
        for j, (prm, gr) in enumerate(zip(self.params, gW + gb)):
            self.m[j] = 0.9 * self.m[j] + 0.1 * gr
            self.v[j] = 0.999 * self.v[j] + 0.001 * gr * gr
            mh = self.m[j] / (1 - 0.9 ** self.t)
            vh = self.v[j] / (1 - 0.999 ** self.t)
            prm -= (LR * mh / (np.sqrt(vh) + 1e-8)).astype(np.float32)

    def predict(self, X):
        return self.forward(X)[-1].argmax(1)

    def qant_path(self, X, variant):
        """linear_fprop -> add_bias_fprop -> relu_fprop ...; softmax_fprop -> argmax, all in bf16."""
        z = bf16(X)
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            Wb = bf16(W)
            o = seq_dot(z, Wb) if variant == "seq" else z @ Wb.T
            z = bf16(bf16(o) + bf16(b))
            if i < len(self.W) - 1:
                z = np.maximum(z, 0)
        e = np.exp(z - z.max(1, keepdims=True))
        return bf16(e / e.sum(1, keepdims=True)).argmax(1)


def job(args):
    dataset, topo, seed = args
    widths = SETS[dataset]["topologies"][topo]
    x_tr, y_tr, x_te, y_te = load(dataset)
    model = MLP(seed, widths)
    rng = np.random.default_rng(seed + 1)
    for _ in range(EPOCHS):
        order = rng.permutation(len(x_tr))
        for s in range(0, len(order), BATCH):
            idx = order[s:s + BATCH]
            model.train_batch(x_tr[idx], y_tr[idx])
    ref = model.predict(x_te)
    row = {"dataset": dataset, "topology": topo, "seed": seed, "reference_accuracy": float((ref == y_te).mean())}
    for v in ("f32acc", "seq"):
        q = np.concatenate([model.qant_path(x_te[s:s + 250], v) for s in range(0, len(x_te), 250)])
        row[f"{v}_disagreements"] = int((q != ref).sum())
        row[f"{v}_accuracy"] = float((q == y_te).mean())
    return row


def run(processes):
    for d in SETS:
        load(d)
    lib()
    jobs = [(d, t, s) for d in SETS for t in SETS[d]["topologies"] for s in SEEDS]
    with mp.get_context("fork").Pool(processes) as pool:
        rows = pool.map(job, jobs)
    summary = []
    for d, cfg in SETS.items():
        arch = json.loads((ROOT / "results" / f"{cfg['archive']}.json").read_text())["results"]
        for t in cfg["topologies"]:
            rs = [r for r in rows if r["dataset"] == d and r["topology"] == t]
            ar = [r for r in arch if r["candidate_id"] == t]
            summary.append({"dataset": d, "topology": t,
                            "reference_accuracy": statistics.fmean(r["reference_accuracy"] for r in rs),
                            "archived_reference_accuracy": statistics.fmean(r["reference_accuracy"] for r in ar),
                            "archived_qant_disagreements": statistics.fmean(r["prediction_disagreements"] for r in ar),
                            "f32acc_disagreements": statistics.fmean(r["f32acc_disagreements"] for r in rs),
                            "seq_disagreements": statistics.fmean(r["seq_disagreements"] for r in rs),
                            "f32acc_accuracy": statistics.fmean(r["f32acc_accuracy"] for r in rs),
                            "seq_accuracy": statistics.fmean(r["seq_accuracy"] for r in rs)})
    return {"experiment": "CL-68", "seeds": list(SEEDS), "epochs": EPOCHS, "batch": BATCH, "lr": LR,
            "cflags": CFLAGS, "numpy": np.__version__,
            "archives": {d: f"results/{c['archive']}.json" for d, c in SETS.items()},
            "rows": rows, "summary": summary}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processes", type=int, default=4)
    args = ap.parse_args()
    res = run(args.processes)
    (HERE / "results.json").write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")
    for s in res["summary"]:
        print(f"{s['dataset']:12s} {s['topology']:9s} ref {s['reference_accuracy']:.4f} (arch {s['archived_reference_accuracy']:.4f})"
              f"  Q.ANT {s['archived_qant_disagreements']:6.1f}  f32acc {s['f32acc_disagreements']:6.1f}  seq {s['seq_disagreements']:7.1f}")


if __name__ == "__main__":
    main()
