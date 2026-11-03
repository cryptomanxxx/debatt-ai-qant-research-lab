"""CL-55: H2 catastrophic forgetting on Permuted MNIST (Tier 1, MNIST reused).

Five tasks learned in sequence (task 1 = original pixel order, tasks 2-5 =
fixed random pixel permutations), shared 10-class head, no task identity.
Single-threaded NumPy per process; seeds run in parallel processes.
See README.md and PREDICTIONS.md.

Run from the repository root:
    python -m experiments.cl55_h2_permuted_mnist_forgetting.run [--data-dir DIR]
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
TASKS = 5
BATCH, LR = 64, 0.1
D, C = cl53.D, cl53.C
REPLAY_PER_TASK = 100
FISHER_SAMPLES = 1000

ARMS = {
    "joint_dense_H256": dict(H=256, mode="joint"),
    "sgd_dense_H256": dict(H=256),
    "sgd_dense_H1024": dict(H=1024),
    "kwta_H1024_k64": dict(H=1024, k=64),
    "fly_frozen_H2048_k64": dict(H=2048, k=64, rule="frozen"),
    "dfa_dense_H256": dict(H=256, rule="dfa"),
    "ewc_dense_H256_lam100": dict(H=256, ewc=100.0),
    "ewc_dense_H256_lam1000": dict(H=256, ewc=1000.0),
    "ewc_dense_H256_lam10000": dict(H=256, ewc=10000.0),
    "replay_dense_H256": dict(H=256, replay=True),
}
PARAMS = ("W1", "b1", "W2", "b2")


class Net:
    """784 -> H -> 10 ReLU MLP; optional kWTA; backprop, DFA or frozen hidden layer."""

    def __init__(self, seed, H, k=None, rule="backprop", ewc=0.0):
        rng = np.random.default_rng(seed)
        self.H, self.k, self.rule, self.lam = H, k, rule, ewc
        self.W1 = (rng.standard_normal((H, D)) * np.sqrt(2 / D)).astype(np.float32)
        self.b1 = np.zeros(H, np.float32)
        self.W2 = (rng.standard_normal((C, H)) * np.sqrt(2 / H)).astype(np.float32)
        self.b2 = np.zeros(C, np.float32)
        self.B = (rng.standard_normal((C, H)) * np.sqrt(2 / H)).astype(np.float32)  # DFA feedback
        self.F = {p: np.zeros_like(getattr(self, p)) for p in PARAMS}   # summed Fisher
        self.FT = {p: np.zeros_like(getattr(self, p)) for p in PARAMS}  # summed Fisher * anchor

    def forward(self, X):
        pre = X @ self.W1.T + self.b1
        h = np.maximum(pre, 0)
        mask = None
        if self.k is not None:
            win = np.argpartition(-pre, self.k - 1, axis=1)[:, :self.k]
            mask = np.zeros(pre.shape, bool)
            np.put_along_axis(mask, win, True, 1)
            h = h * mask
        return pre, mask, h, h @ self.W2.T + self.b2

    def _grads(self, X, y, mean=True):
        pre, mask, h, logits = self.forward(X)
        logits = logits - logits.max(1, keepdims=True)
        p = np.exp(logits)
        p /= p.sum(1, keepdims=True)
        p[np.arange(len(y)), y] -= 1
        g = p / len(y) if mean else p
        fb = self.B if self.rule == "dfa" else self.W2
        dh = (g @ fb) * (pre > 0)
        if mask is not None:
            dh *= mask
        return g, h, dh

    def train_batch(self, X, y):
        g, h, dh = self._grads(X, y)
        grads = {"W2": g.T @ h, "b2": g.sum(0)}
        if self.rule != "frozen":
            grads["W1"] = dh.T @ X
            grads["b1"] = dh.sum(0)
        for name, grad in grads.items():
            param = getattr(self, name)
            param -= LR * grad
            if self.lam:
                # Proximal (implicit) step for the EWC penalty (lam/2) * sum F (theta - anchor)^2:
                # exactly minimises ||theta - theta'||^2 / (2 LR) + penalty, stable for any lam*F.
                # The explicit gradient step diverged when LR * lam * F > 2.
                c = LR * self.lam
                param[...] = (param + c * self.FT[name]) / (1 + c * self.F[name])

    def consolidate(self, X, y):
        """Add the diagonal empirical Fisher of the current task (EWC anchor)."""
        g, h, dh = self._grads(X, y, mean=False)
        n = len(y)
        fisher = {"W2": (g * g).T @ (h * h) / n, "b2": (g * g).mean(0),
                  "W1": (dh * dh).T @ (X * X) / n, "b1": (dh * dh).mean(0)}
        for name in PARAMS:
            self.F[name] += fisher[name]
            self.FT[name] += fisher[name] * getattr(self, name)

    def accuracy(self, X, y, batch=2000):
        correct = 0
        for s in range(0, len(X), batch):
            correct += int((self.forward(X[s:s + batch])[3].argmax(1) == y[s:s + batch]).sum())
        return correct / len(X)


def permutations(seed):
    rng = np.random.default_rng(seed + 100)
    return [np.arange(D)] + [rng.permutation(D) for _ in range(TASKS - 1)]


def run_seed(args):
    seed, data_dir, arms, n_train = args
    x_tr, y_tr, x_te, y_te = cl53.load_mnist(Path(data_dir))
    if n_train:
        x_tr, y_tr = x_tr[:n_train], y_tr[:n_train]
    perms = permutations(seed)
    tests = [(np.ascontiguousarray(x_te[:, p]), y_te) for p in perms]
    out = {}
    for arm, cfg in arms.items():
        cfg = dict(cfg)
        mode = cfg.pop("mode", "sequential")
        replay = cfg.pop("replay", False)
        net = Net(seed, **cfg)
        start = time.perf_counter()
        if mode == "joint":
            X = np.concatenate([x_tr[:, p] for p in perms])
            Y = np.concatenate([y_tr] * TASKS)
            order = np.random.default_rng(seed + 1).permutation(len(X))
            for s in range(0, len(order), BATCH):
                idx = order[s:s + BATCH]
                net.train_batch(X[idx], Y[idx])
            matrix = [[net.accuracy(*t) for t in tests]]
        else:
            matrix, buf_x, buf_y = [], [], []
            for t, p in enumerate(perms):
                Xt = np.ascontiguousarray(x_tr[:, p])
                rng = np.random.default_rng(seed * 100 + t)
                order = rng.permutation(len(Xt))
                for s in range(0, len(order), BATCH):
                    idx = order[s:s + BATCH]
                    xb, yb = Xt[idx], y_tr[idx]
                    if replay and buf_x:
                        bx, by = np.concatenate(buf_x), np.concatenate(buf_y)
                        r = rng.integers(0, len(bx), BATCH)
                        xb, yb = np.concatenate([xb, bx[r]]), np.concatenate([yb, by[r]])
                    net.train_batch(xb, yb)
                if replay:
                    keep = rng.choice(len(Xt), REPLAY_PER_TASK, replace=False)
                    buf_x.append(Xt[keep])
                    buf_y.append(y_tr[keep])
                if net.lam:
                    f = rng.choice(len(Xt), FISHER_SAMPLES, replace=False)
                    net.consolidate(Xt[f], y_tr[f])
                matrix.append([net.accuracy(*tt) for tt in tests])
        out[arm] = {"matrix": matrix, "seconds": time.perf_counter() - start}
    return seed, out


def metrics(matrix):
    final = matrix[-1]
    acc = sum(final) / len(final)
    if len(matrix) == 1:
        return {"ACC": acc, "BWT": None, "LA": None}
    T = len(matrix)
    return {"ACC": acc,
            "BWT": sum(final[i] - matrix[i][i] for i in range(T - 1)) / (T - 1),
            "LA": sum(matrix[i][i] for i in range(T)) / T}


def summarize(per_seed):
    keys = ("ACC", "BWT", "LA")
    out = {}
    for key in keys:
        vals = [m[key] for m in per_seed if m[key] is not None]
        if vals:
            out[key] = {"mean": statistics.fmean(vals), "sd": statistics.stdev(vals) if len(vals) > 1 else 0.0,
                        "per_seed": vals}
    return out


def run(data_dir: Path, seeds=SEEDS, arms=ARMS, n_train=None, processes=3):
    jobs = [(s, str(data_dir), arms, n_train) for s in seeds]
    if processes > 1:
        with mp.get_context("fork").Pool(min(processes, len(seeds))) as pool:
            results = dict(pool.map(run_seed, jobs))
    else:
        results = dict(map(run_seed, jobs))
    out = {"experiment": "CL-55", "dataset": "Permuted MNIST", "tasks": TASKS, "seeds": list(seeds),
           "epochs_per_task": 1, "batch": BATCH, "lr": LR, "replay_per_task": REPLAY_PER_TASK,
           "fisher_samples": FISHER_SAMPLES, "mnist_sha256": cl53.FILES, "numpy": np.__version__,
           "arms": {}}
    timing = {}
    for arm in arms:
        per_seed = [metrics(results[s][arm]["matrix"]) for s in seeds]
        out["arms"][arm] = {"config": arms[arm], **summarize(per_seed),
                            "accuracy_matrix_per_seed": {str(s): results[s][arm]["matrix"] for s in seeds}}
        timing[arm] = {str(s): results[s][arm]["seconds"] for s in seeds}
    return out, timing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=HERE.parent / "cl53_mnist_hidden_sparsity" / "data")
    args = ap.parse_args()
    results, timing = run(args.data_dir)
    (HERE / "results.json").write_text(json.dumps(results, indent=1, sort_keys=True) + "\n")
    (HERE / "timing.json").write_text(json.dumps(timing, indent=1, sort_keys=True) + "\n")
    for arm, r in results["arms"].items():
        parts = [f"{k} {r[k]['mean']:+.4f} (sd {r[k]['sd']:.4f})" for k in ("ACC", "BWT", "LA") if k in r]
        print(f"{arm:26s} " + "  ".join(parts))


if __name__ == "__main__":
    main()
