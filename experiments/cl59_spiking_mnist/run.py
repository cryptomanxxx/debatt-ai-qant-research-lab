"""CL-59: a spiking (LIF) network on MNIST against a dense ReLU network (Tier 1).

Rate-coded input, LIF hidden layer trained with BPTT and a surrogate gradient,
non-spiking readout averaged over T steps. Single-threaded NumPy; jobs run in
parallel processes. Reuses the CL-53 MNIST download. See README.md and
PREDICTIONS.md.

Run from the repository root:
    python -m experiments.cl59_spiking_mnist.run [--data-dir DIR]
"""
from __future__ import annotations

import experiments.cl53_mnist_hidden_sparsity.run as cl53  # sets single-threaded BLAS first

import argparse
import json
import multiprocessing as mp
import statistics
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SEEDS = (11, 22, 33)
ARMS = {"ann_dense_H256": None, "snn_T4": 4, "snn_T8": 8, "snn_T16": 16}
H, D, C = 256, cl53.D, cl53.C
EPOCHS, BATCH, LR = 2, 64, 1e-3
BETA, THETA, SLOPE = 0.9, 1.0, 5.0


class Adam:
    def __init__(self, params):
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, params, grads):
        self.t += 1
        for k, g in grads.items():
            self.m[k] = 0.9 * self.m[k] + 0.1 * g
            self.v[k] = 0.999 * self.v[k] + 0.001 * g * g
            mh = self.m[k] / (1 - 0.9 ** self.t)
            vh = self.v[k] / (1 - 0.999 ** self.t)
            params[k] -= (LR * mh / (np.sqrt(vh) + 1e-8)).astype(np.float32)


def init_params(seed):
    rng = np.random.default_rng(seed)
    return {"W1": (rng.standard_normal((H, D)) * np.sqrt(2 / D)).astype(np.float32),
            "b1": np.zeros(H, np.float32),
            "W2": (rng.standard_normal((C, H)) * np.sqrt(2 / H)).astype(np.float32),
            "b2": np.zeros(C, np.float32)}


def softmax_grad(logits, y):
    z = logits - logits.max(1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(1, keepdims=True)
    p[np.arange(len(y)), y] -= 1
    return (p / len(y)).astype(np.float32)


class ANN:
    def __init__(self, seed):
        self.p = init_params(seed)
        self.opt = Adam(self.p)

    def logits(self, X):
        h = np.maximum(X @ self.p["W1"].T + self.p["b1"], 0)
        return h, h @ self.p["W2"].T + self.p["b2"]

    def grads(self, X, y):
        h, logits = self.logits(X)
        g = softmax_grad(logits, y)
        dh = (g @ self.p["W2"]) * (h > 0)
        return {"W1": dh.T @ X, "b1": dh.sum(0), "W2": g.T @ h, "b2": g.sum(0)}

    def train_batch(self, X, y, rng):
        self.opt.step(self.p, self.grads(X, y))

    def predict(self, X, rng=None):
        return self.logits(X)[1].argmax(1), None


class SNN:
    """784 -> H LIF (reset by subtraction) -> 10 readout averaged over T steps."""

    def __init__(self, seed, T, smooth=False):
        self.p = init_params(seed)
        self.opt = Adam(self.p)
        self.T, self.smooth = T, smooth

    def _spike(self, v):
        x = v - THETA
        if self.smooth:  # differentiable stand-in, used only by the gradient test
            s = 1 / (1 + np.exp(-SLOPE * x))
            return s, SLOPE * s * (1 - s)
        return (x >= 0).astype(np.float32), 1 / (1 + SLOPE * np.abs(x)) ** 2

    def run(self, S):
        """S: (T, B, D) input spikes. Returns per-step v, s, surrogate grads and logits."""
        T, B = S.shape[0], S.shape[1]
        v = np.zeros((B, H), np.float32)
        s = np.zeros((B, H), np.float32)
        vs, ss, ds = [], [], []
        for t in range(T):
            v = BETA * v + S[t] @ self.p["W1"].T + self.p["b1"] - THETA * s
            s, d = self._spike(v)
            vs.append(v)
            ss.append(s)
            ds.append(d)
        s_mean = sum(ss) / T
        return ss, ds, s_mean, s_mean @ self.p["W2"].T + self.p["b2"]

    def train_batch(self, X, y, rng):
        S = (rng.random((self.T,) + X.shape) < X).astype(np.float32)
        self.opt.step(self.p, self.grads(S, y))

    def grads(self, S, y):
        ss, ds, s_mean, logits = self.run(S)
        g = softmax_grad(logits, y)
        ds_out = (g @ self.p["W2"]) / self.T
        dW1 = np.zeros_like(self.p["W1"])
        db1 = np.zeros_like(self.p["b1"])
        dv_next = np.zeros_like(ds_out)
        for t in range(self.T - 1, -1, -1):
            # v_{t+1} = beta v_t + I_{t+1} - theta s_t ; s_t = spike(v_t)
            dv = (ds_out - THETA * dv_next) * ds[t] + BETA * dv_next
            dW1 += dv.T @ S[t]
            db1 += dv.sum(0)
            dv_next = dv
        return {"W1": dW1, "b1": db1, "W2": g.T @ s_mean, "b2": g.sum(0)}

    def predict(self, X, rng):
        S = (rng.random((self.T,) + X.shape) < X).astype(np.float32)
        ss, _, _, logits = self.run(S)
        return logits.argmax(1), (S.sum(), sum(s.sum() for s in ss))


def run_job(args):
    arm, seed, data_dir, n_train = args
    x_tr, y_tr, x_te, y_te = cl53.load_mnist(Path(data_dir))
    if n_train:
        x_tr, y_tr = x_tr[:n_train], y_tr[:n_train]
    T = ARMS[arm]
    model = ANN(seed) if T is None else SNN(seed, T)
    rng = np.random.default_rng(seed + 1)
    epoch_seconds = []
    for _ in range(EPOCHS):
        start = time.perf_counter()
        order = rng.permutation(len(x_tr))
        for s in range(0, len(order), BATCH):
            idx = order[s:s + BATCH]
            model.train_batch(x_tr[idx], y_tr[idx], rng)
        epoch_seconds.append(time.perf_counter() - start)
    eval_rng = np.random.default_rng(seed + 999)
    correct, in_spikes, hid_spikes = 0, 0.0, 0.0
    for s in range(0, len(x_te), 500):
        pred, spikes = model.predict(x_te[s:s + 500], eval_rng)
        correct += int((pred == y_te[s:s + 500]).sum())
        if spikes:
            in_spikes += float(spikes[0])
            hid_spikes += float(spikes[1])
    n = len(x_te)
    row = {"arm": arm, "seed": seed, "test_accuracy": correct / n, "epoch_seconds_mean": statistics.fmean(epoch_seconds)}
    if T is None:
        nnz = float((x_te != 0).sum(1).mean())
        row.update({"dense_macs": D * H + H * C, "zero_skip_macs": nnz * H + H * C})
    else:
        row.update({"T": T,
                    "input_spikes_per_image": in_spikes / n,
                    "hidden_spikes_per_image": hid_spikes / n,
                    "hidden_spike_rate": hid_spikes / (n * H * T),
                    "synaptic_ops_per_image": (in_spikes * H + hid_spikes * C) / n,
                    "input_layer_ops_per_image": in_spikes * H / n,
                    "analytic_input_layer_ops": T * float(x_te.sum(1).mean()) * H,
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
    return {"experiment": "CL-59", "dataset": "MNIST", "mnist_sha256": cl53.FILES, "seeds": list(seeds), "H": H,
            "epochs": EPOCHS, "batch": BATCH, "lr": LR, "beta": BETA, "theta": THETA, "surrogate_slope": SLOPE,
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
            line += (f" SOPs {s['synaptic_ops_per_image']['mean']:,.0f} (input {s['input_layer_ops_per_image']['mean']:,.0f},"
                     f" analytic {s['analytic_input_layer_ops']['mean']:,.0f}) hidden rate {s['hidden_spike_rate']['mean']:.3f}")
        else:
            line += f" MACs dense {s['dense_macs']['mean']:,.0f} zero-skip {s['zero_skip_macs']['mean']:,.0f}"
        print(line)


if __name__ == "__main__":
    main()
