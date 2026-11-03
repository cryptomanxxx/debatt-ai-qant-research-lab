"""CL-61: spike-rate penalty for the latency-coded SNN, against smaller dense nets (Tier 1).

Loss = cross-entropy + lam * hidden spike rate (spikes / (H * T), batch mean).
Latency coding and LIF dynamics as in CL-60. Dense baselines use zero-skipping
MAC counts (CL-57). See README.md and PREDICTIONS.md.

Run from the repository root:
    python -m experiments.cl61_spike_rate_penalty.run [--data-dir DIR]
"""
from __future__ import annotations

import experiments.cl60_latency_coded_snn.run as cl60  # sets single-threaded BLAS first

import argparse
import json
import multiprocessing as mp
import statistics
import time
from pathlib import Path

import numpy as np

cl59, cl53 = cl60.cl59, cl60.cl53
HERE = Path(__file__).resolve().parent
SEEDS = cl59.SEEDS
D, C, T = cl59.D, cl59.C, 4
LAMBDAS = (0.0, 0.03, 0.1, 0.3, 1.0, 3.0)
ARMS = {f"ann_H{h}": ("ann", h, None) for h in (64, 128, 256)}
ARMS.update({f"lat_T4_H256_lam{lam:g}": ("snn", 256, lam) for lam in LAMBDAS})
ARMS["lat_T4_H64_lam0"] = ("snn", 64, 0.0)


def init_params(seed, H):
    rng = np.random.default_rng(seed)
    return {"W1": (rng.standard_normal((H, D)) * np.sqrt(2 / D)).astype(np.float32),
            "b1": np.zeros(H, np.float32),
            "W2": (rng.standard_normal((C, H)) * np.sqrt(2 / H)).astype(np.float32),
            "b2": np.zeros(C, np.float32)}


class ANN(cl59.ANN):
    def __init__(self, seed, H):
        self.p = init_params(seed, H)
        self.opt = cl59.Adam(self.p)


class PenalizedSNN(cl60.LatencySNN):
    def __init__(self, seed, H, lam, T=T, smooth=False):
        self.p = init_params(seed, H)
        self.opt = cl59.Adam(self.p)
        self.H, self.lam, self.T, self.smooth = H, lam, T, smooth

    def run(self, S):
        B = S.shape[1]
        v = np.zeros((B, self.H), np.float32)
        s = np.zeros((B, self.H), np.float32)
        ss, ds = [], []
        for t in range(S.shape[0]):
            v = cl59.BETA * v + S[t] @ self.p["W1"].T + self.p["b1"] - cl59.THETA * s
            s, d = self._spike(v)
            ss.append(s)
            ds.append(d)
        s_mean = sum(ss) / len(ss)
        return ss, ds, s_mean, s_mean @ self.p["W2"].T + self.p["b2"]

    def loss(self, S, y):
        """Cross-entropy + lam * spike rate; used by the gradient test."""
        ss, _, _, logits = self.run(S)
        z = logits - logits.max(1, keepdims=True)
        ce = -(z[np.arange(len(y)), y] - np.log(np.exp(z).sum(1))).mean()
        return ce + self.lam * float(np.mean(ss))

    def grads(self, S, y):
        ss, ds, s_mean, logits = self.run(S)
        Tn, B = S.shape[0], S.shape[1]
        g = cl59.softmax_grad(logits, y)
        ds_out = (g @ self.p["W2"]) / Tn + np.float32(self.lam / (B * self.H * Tn))
        dW1 = np.zeros_like(self.p["W1"])
        db1 = np.zeros_like(self.p["b1"])
        dv_next = np.zeros_like(ds_out)
        for t in range(Tn - 1, -1, -1):
            dv = (ds_out - cl59.THETA * dv_next) * ds[t] + cl59.BETA * dv_next
            dW1 += dv.T @ S[t]
            db1 += dv.sum(0)
            dv_next = dv
        return {"W1": dW1, "b1": db1, "W2": g.T @ s_mean, "b2": g.sum(0)}


def run_job(args):
    arm, seed, data_dir = args
    x_tr, y_tr, x_te, y_te = cl53.load_mnist(Path(data_dir))
    kind, H, lam = ARMS[arm]
    model = ANN(seed, H) if kind == "ann" else PenalizedSNN(seed, H, lam)
    rng = np.random.default_rng(seed + 1)
    epoch_seconds = []
    for _ in range(cl59.EPOCHS):
        start = time.perf_counter()
        order = rng.permutation(len(x_tr))
        for s in range(0, len(order), cl59.BATCH):
            idx = order[s:s + cl59.BATCH]
            model.train_batch(x_tr[idx], y_tr[idx], rng)
        epoch_seconds.append(time.perf_counter() - start)
    correct, in_spikes, hid_spikes = 0, 0.0, 0.0
    for s in range(0, len(x_te), 500):
        pred, spikes = model.predict(x_te[s:s + 500])
        correct += int((pred == y_te[s:s + 500]).sum())
        if spikes:
            in_spikes += float(spikes[0])
            hid_spikes += float(spikes[1])
    n = len(x_te)
    nnz = float((x_te != 0).sum(1).mean())
    row = {"arm": arm, "seed": seed, "H": H, "test_accuracy": correct / n,
           "epoch_seconds_mean": statistics.fmean(epoch_seconds)}
    if kind == "ann":
        row.update({"dense_macs": D * H + H * C, "ops_per_image": nnz * H + H * C})
    else:
        row.update({"lam": lam, "T": T,
                    "input_spikes_per_image": in_spikes / n,
                    "hidden_spikes_per_image": hid_spikes / n,
                    "hidden_spike_rate": hid_spikes / (n * H * T),
                    "ops_per_image": (in_spikes * H + hid_spikes * C) / n,
                    "input_layer_ops_per_image": in_spikes * H / n})
    return row


def run(data_dir: Path, seeds=SEEDS, arms=ARMS, processes=3):
    cl53.load_mnist(data_dir)
    jobs = [(a, s, str(data_dir)) for a in arms for s in seeds]
    if processes > 1:
        with mp.get_context("fork").Pool(processes) as pool:
            rows = pool.map(run_job, jobs)
    else:
        rows = list(map(run_job, jobs))
    summary = {}
    for a in arms:
        rs = [r for r in rows if r["arm"] == a]
        keys = [k for k in rs[0] if k not in ("arm", "seed")]
        summary[a] = {k: {"mean": statistics.fmean(r[k] for r in rs), "per_seed": [r[k] for r in rs]} for k in keys}
    return {"experiment": "CL-61", "dataset": "MNIST", "mnist_sha256": cl53.FILES, "seeds": list(seeds), "T": T,
            "lambdas": list(LAMBDAS), "epochs": cl59.EPOCHS, "batch": cl59.BATCH, "lr": cl59.LR, "beta": cl59.BETA,
            "theta": cl59.THETA, "surrogate_slope": cl59.SLOPE,
            "penalty": "lam * hidden spikes / (H * T), batch mean", "numpy": np.__version__,
            "rows": rows, "summary": summary}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=HERE.parent / "cl53_mnist_hidden_sparsity" / "data")
    ap.add_argument("--processes", type=int, default=3)
    args = ap.parse_args()
    results = run(args.data_dir, processes=args.processes)
    (HERE / "results.json").write_text(json.dumps(results, indent=1, sort_keys=True) + "\n")
    for a, s in results["summary"].items():
        line = (f"{a:20s} acc {100 * s['test_accuracy']['mean']:.2f}% {[round(100 * x, 2) for x in s['test_accuracy']['per_seed']]}"
                f" ops {s['ops_per_image']['mean']:,.0f} epoch {s['epoch_seconds_mean']['mean']:.1f}s")
        if "hidden_spikes_per_image" in s:
            line += f" hidden spikes {s['hidden_spikes_per_image']['mean']:.1f} rate {s['hidden_spike_rate']['mean']:.3f}"
        print(line)


if __name__ == "__main__":
    main()
