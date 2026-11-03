"""CL-60: a latency-coded spiking (LIF) network on MNIST (Tier 1).

Same network, training and counts as CL-59; only the input encoding changes.
Each nonzero pixel spikes exactly once, at step round((1 - x)(T - 1)), and zero
pixels never spike. See README.md and PREDICTIONS.md.

Run from the repository root:
    python -m experiments.cl60_latency_coded_snn.run [--data-dir DIR]
"""
from __future__ import annotations

import experiments.cl59_spiking_mnist.run as cl59  # sets single-threaded BLAS first

import argparse
import json
import multiprocessing as mp
import statistics
import time
from pathlib import Path

import numpy as np

cl53 = cl59.cl53
HERE = Path(__file__).resolve().parent
SEEDS = cl59.SEEDS
ARMS = {"ann_dense_H256": None, "lat_T4": 4, "lat_T8": 8, "lat_T16": 16}
H, D, C = cl59.H, cl59.D, cl59.C


def latency_encode(X, T):
    """(B, D) intensities in [0, 1] -> (T, B, D) spikes, one per nonzero pixel."""
    t_idx = np.rint((1 - X) * (T - 1)).astype(np.int64)
    return ((np.arange(T)[:, None, None] == t_idx[None]) & (X > 0)[None]).astype(np.float32)


class LatencySNN(cl59.SNN):
    def train_batch(self, X, y, rng):
        self.opt.step(self.p, self.grads(latency_encode(X, self.T), y))

    def predict(self, X, rng=None):
        S = latency_encode(X, self.T)
        ss, _, _, logits = self.run(S)
        return logits.argmax(1), (S.sum(), sum(s.sum() for s in ss))


def run_job(args):
    arm, seed, data_dir, n_train = args
    x_tr, y_tr, x_te, y_te = cl53.load_mnist(Path(data_dir))
    if n_train:
        x_tr, y_tr = x_tr[:n_train], y_tr[:n_train]
    T = ARMS[arm]
    model = cl59.ANN(seed) if T is None else LatencySNN(seed, T)
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
    row = {"arm": arm, "seed": seed, "test_accuracy": correct / n, "epoch_seconds_mean": statistics.fmean(epoch_seconds)}
    nnz = float((x_te != 0).sum(1).mean())
    if T is None:
        row.update({"dense_macs": D * H + H * C, "zero_skip_macs": nnz * H + H * C})
    else:
        row.update({"T": T,
                    "input_spikes_per_image": in_spikes / n,
                    "analytic_input_spikes_per_image": nnz,
                    "hidden_spikes_per_image": hid_spikes / n,
                    "hidden_spike_rate": hid_spikes / (n * H * T),
                    "synaptic_ops_per_image": (in_spikes * H + hid_spikes * C) / n,
                    "input_layer_ops_per_image": in_spikes * H / n,
                    "neuron_updates_per_image": H * T})
    return row


def run(data_dir: Path, seeds=SEEDS, arms=ARMS, n_train=None, processes=3):
    cl53.load_mnist(data_dir)
    jobs = [(a, s, str(data_dir), n_train) for a in arms for s in seeds]
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
    return {"experiment": "CL-60", "dataset": "MNIST", "mnist_sha256": cl53.FILES, "seeds": list(seeds), "H": H,
            "epochs": cl59.EPOCHS, "batch": cl59.BATCH, "lr": cl59.LR, "beta": cl59.BETA, "theta": cl59.THETA,
            "surrogate_slope": cl59.SLOPE, "encoding": "latency: t = round((1 - x)(T - 1)), x > 0 only",
            "numpy": np.__version__, "rows": rows, "summary": summary}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=HERE.parent / "cl53_mnist_hidden_sparsity" / "data")
    ap.add_argument("--processes", type=int, default=3)
    args = ap.parse_args()
    results = run(args.data_dir, processes=args.processes)
    (HERE / "results.json").write_text(json.dumps(results, indent=1, sort_keys=True) + "\n")
    for a, s in results["summary"].items():
        line = f"{a:16s} acc {100 * s['test_accuracy']['mean']:.2f}% {[round(100 * x, 2) for x in s['test_accuracy']['per_seed']]} epoch {s['epoch_seconds_mean']['mean']:.1f}s"
        if "synaptic_ops_per_image" in s:
            line += (f" SOPs {s['synaptic_ops_per_image']['mean']:,.0f} (input spikes {s['input_spikes_per_image']['mean']:.2f},"
                     f" hidden spikes {s['hidden_spikes_per_image']['mean']:.1f}) hidden rate {s['hidden_spike_rate']['mean']:.3f}")
        else:
            line += f" MACs dense {s['dense_macs']['mean']:,.0f} zero-skip {s['zero_skip_macs']['mean']:,.0f}"
        print(line)


if __name__ == "__main__":
    main()
