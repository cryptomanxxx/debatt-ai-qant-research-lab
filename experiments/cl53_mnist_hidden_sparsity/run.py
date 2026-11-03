"""CL-53: hidden-layer kWTA sparsity and cheap winner selection on MNIST (Tier 1).

Downloads MNIST (public dataset, owner-approved) if missing, verifies pinned
SHA-256 digests, and trains small NumPy MLPs single-threaded on CPU.

Run from the repository root:
    python -m experiments.cl53_mnist_hidden_sparsity.run [--data-dir DIR]
"""
from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_var] = "1"  # enforce single-threaded BLAS: reproducible numbers, comparable timing

import argparse
import gzip
import hashlib
import json
import statistics
import time
import urllib.request
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
URL = "https://ossci-datasets.s3.amazonaws.com/mnist/"
FILES = {
    "train-images-idx3-ubyte.gz": "440fcabf73cc546fa21475e81ea370265605f56be210a4024d2ca8f203523609",
    "train-labels-idx1-ubyte.gz": "3552534a0a558bbed6aed32b30c495cca23d567ec52cac8be1a0730e8010255c",
    "t10k-images-idx3-ubyte.gz": "8d422c7b0a1c1c79245a5bcf07fe86e33eeafee792b84584aec276f5a2dbc4e6",
    "t10k-labels-idx1-ubyte.gz": "f7ae60f92e00ec6debd23a6088c31dbd2371eca3ffa0defaefb259924204aec6",
}
SEEDS = (11, 22, 33)
EPOCHS, BATCH, LR = 2, 64, 0.05
D, C = 784, 10
REFRESH_BATCHES = 50  # low-rank selector refresh interval (3,200 samples)
TIMING_REPEATS = 7

ARMS = {
    **{f"dense_H{h}": dict(H=h) for h in (32, 64, 128, 256, 512)},
    "kwta_exact_H1024_k32": dict(H=1024, k=32, sel="exact"),
    "kwta_exact_H1024_k64": dict(H=1024, k=64, sel="exact"),
    "kwta_lowrank_H1024_k32": dict(H=1024, k=32, sel="lowrank", r=32, c=128),
    "kwta_lowrank_H1024_k64": dict(H=1024, k=64, sel="lowrank", r=32, c=256),
}


# ------------------------------------------------------------------ data

def load_mnist(data_dir: Path):
    data_dir.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for name, digest in FILES.items():
        path = data_dir / name
        if not path.exists():
            urllib.request.urlretrieve(URL + name, path)
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != digest:
            raise RuntimeError(f"{name}: SHA-256 {actual} does not match pinned {digest}")
        raw = gzip.decompress(path.read_bytes())
        magic = int.from_bytes(raw[:4], "big")
        if magic == 2051:  # images
            n, rows, cols = (int.from_bytes(raw[4 + 4 * i:8 + 4 * i], "big") for i in range(3))
            arrays[name] = np.frombuffer(raw, np.uint8, offset=16).reshape(n, rows * cols)
        elif magic == 2049:  # labels
            arrays[name] = np.frombuffer(raw, np.uint8, offset=8)
        else:
            raise RuntimeError(f"{name}: unexpected IDX magic {magic}")
    x_tr = arrays["train-images-idx3-ubyte.gz"].astype(np.float32) / 255
    y_tr = arrays["train-labels-idx1-ubyte.gz"].astype(np.int64)
    x_te = arrays["t10k-images-idx3-ubyte.gz"].astype(np.float32) / 255
    y_te = arrays["t10k-labels-idx1-ubyte.gz"].astype(np.int64)
    assert x_tr.shape == (60000, D) and x_te.shape == (10000, D)
    return x_tr, y_tr, x_te, y_te


# ------------------------------------------------------------------ model

class Net:
    def __init__(self, seed, H, k=None, sel=None, r=None, c=None):
        rng = np.random.default_rng(seed)
        self.H, self.k, self.sel, self.r, self.c = H, k, sel, r, c
        self.W1 = (rng.standard_normal((H, D)) * np.sqrt(2 / D)).astype(np.float32)
        self.b1 = np.zeros(H, np.float32)
        self.W2 = (rng.standard_normal((C, H)) * np.sqrt(2 / H)).astype(np.float32)
        self.b2 = np.zeros(C, np.float32)
        self.refresh_terms = 0
        if sel == "lowrank":
            self.V = rng.standard_normal((D, r)).astype(np.float32)
            self.refresh(iters=4)

    def refresh(self, iters=2):
        """Subspace iteration for the top-r right singular vectors of W1."""
        V = self.V
        for _ in range(iters):
            V, _ = np.linalg.qr(self.W1.T @ (self.W1 @ V))
        self.V = V.astype(np.float32)
        self.A = (self.W1 @ self.V).astype(np.float32)
        self.refresh_terms += iters * 2 * self.H * D * self.r + self.H * D * self.r

    def winners_mask(self, X, pre):
        """Boolean mask of the k winners per row (None = dense)."""
        if self.k is None:
            return None
        if self.sel == "exact":
            cand_pre = pre
        else:
            score = (X @ self.V) @ self.A.T + self.b1
            cand = np.argpartition(-score, self.c - 1, axis=1)[:, :self.c]
            cand_pre = np.full_like(pre, -np.inf)
            np.put_along_axis(cand_pre, cand, np.take_along_axis(pre, cand, 1), 1)
        win = np.argpartition(-cand_pre, self.k - 1, axis=1)[:, :self.k]
        mask = np.zeros(pre.shape, bool)
        np.put_along_axis(mask, win, True, 1)
        return mask

    def forward(self, X):
        pre = X @ self.W1.T + self.b1  # numerically equal to computing only candidates
        mask = self.winners_mask(X, pre)
        h = np.maximum(pre, 0)
        if mask is not None:
            h = h * mask
        return pre, mask, h, h @ self.W2.T + self.b2

    def train_batch(self, X, y):
        pre, mask, h, logits = self.forward(X)
        logits = logits - logits.max(1, keepdims=True)
        p = np.exp(logits)
        p /= p.sum(1, keepdims=True)
        p[np.arange(len(y)), y] -= 1
        g = p / len(y)
        dh = (g @ self.W2) * (pre > 0)
        if mask is not None:
            dh *= mask
        self.W2 -= LR * (g.T @ h)
        self.b2 -= LR * g.sum(0)
        self.W1 -= LR * (dh.T @ X)
        self.b1 -= LR * dh.sum(0)

    def predict(self, X):
        return self.forward(X)[3].argmax(1)

    # Analytic forward work per inference sample (dense fan-in D).
    def inference_terms(self):
        if self.k is None:
            return {"select": 0, "hidden": D * self.H, "out": C * self.H}
        if self.sel == "exact":
            return {"select": self.H, "hidden": D * self.H, "out": C * self.k}
        return {"select": D * self.r + self.H * self.r + self.H + self.c,
                "hidden": D * self.c, "out": C * self.k}

    # Genuinely sparse inference path, used only for wall-time measurement.
    def predict_sparse(self, X, batch=256):
        out = np.empty(len(X), np.int64)
        for s in range(0, len(X), batch):
            xb = X[s:s + batch]
            if self.k is None or self.sel == "exact":
                out[s:s + batch] = self.predict(xb)
                continue
            score = (xb @ self.V) @ self.A.T + self.b1
            cand = np.argpartition(-score, self.c - 1, axis=1)[:, :self.c]
            pre_c = np.einsum("bcd,bd->bc", self.W1[cand], xb) + self.b1[cand]
            top = np.argpartition(-pre_c, self.k - 1, axis=1)[:, :self.k]
            win = np.take_along_axis(cand, top, 1)
            h = np.maximum(np.take_along_axis(pre_c, top, 1), 0)
            logits = np.einsum("bkc,bk->bc", self.W2.T[win], h) + self.b2
            out[s:s + batch] = logits.argmax(1)
        return out


def recall(net, X):
    pre = X @ net.W1.T + net.b1
    exact = np.argpartition(-pre, net.k - 1, axis=1)[:, :net.k]
    em = np.zeros(pre.shape, bool)
    np.put_along_axis(em, exact, True, 1)
    return float((net.winners_mask(X, pre) & em).sum() / em.sum())


def median_time(fn, repeats=TIMING_REPEATS):
    times = []
    for _ in range(repeats):
        t = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t)
    return statistics.median(times), min(times), max(times)


# ------------------------------------------------------------------ run

def run(data_dir: Path, seeds=SEEDS, epochs=EPOCHS, arms=ARMS, n_train=None, timing=True):
    x_tr, y_tr, x_te, y_te = load_mnist(data_dir)
    if n_train:
        x_tr, y_tr = x_tr[:n_train], y_tr[:n_train]
    out = {"experiment": "CL-53", "dataset": "MNIST", "sha256": FILES, "seeds": list(seeds),
           "epochs": epochs, "batch": BATCH, "lr": LR, "numpy": np.__version__,
           "threads": os.environ.get("OPENBLAS_NUM_THREADS"), "arms": {}}
    timing_out = {}
    for arm, cfg in arms.items():
        accs, recalls, train_refresh = [], [], []
        last = None
        for seed in seeds:
            net = Net(seed, **cfg)
            rng = np.random.default_rng(seed + 1)
            batches = 0
            for _ in range(epochs):
                order = rng.permutation(len(x_tr))
                for s in range(0, len(order), BATCH):
                    idx = order[s:s + BATCH]
                    net.train_batch(x_tr[idx], y_tr[idx])
                    batches += 1
                    if net.sel == "lowrank" and batches % REFRESH_BATCHES == 0:
                        net.refresh()
            accs.append(float((net.predict(x_te) == y_te).mean()))
            if net.k is not None:
                recalls.append(recall(net, x_te))
            train_refresh.append(net.refresh_terms / (epochs * len(x_tr)))
            last = net
        terms = last.inference_terms()
        out["arms"][arm] = {
            "config": cfg, "test_accuracy_per_seed": accs,
            "test_accuracy_mean": statistics.fmean(accs),
            "test_accuracy_sd": statistics.stdev(accs) if len(accs) > 1 else 0.0,
            "top_k_recall_mean": statistics.fmean(recalls) if recalls else None,
            "inference_terms_per_sample": terms,
            "inference_terms_total": sum(terms.values()),
            "selector_refresh_terms_per_training_sample": statistics.fmean(train_refresh),
        }
        if timing:
            assert (last.predict_sparse(x_te) == last.predict(x_te)).mean() > 0.999
            timing_out[arm] = median_time(lambda: last.predict_sparse(x_te))
            if arm == "dense_H256":
                timing_out["dense_H256_AA"] = median_time(lambda: last.predict_sparse(x_te))
    return out, timing_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=HERE / "data")
    args = ap.parse_args()
    results, timing = run(args.data_dir)
    (HERE / "results.json").write_text(json.dumps(results, indent=1, sort_keys=True) + "\n")
    (HERE / "timing.json").write_text(json.dumps(
        {k: {"median_s": v[0], "min_s": v[1], "max_s": v[2]} for k, v in timing.items()},
        indent=1, sort_keys=True) + "\n")
    ref = results["arms"]["dense_H256"]["inference_terms_total"]
    tref = timing["dense_H256"][0]
    for arm, r in results["arms"].items():
        rc = r["top_k_recall_mean"]
        print(f"{arm:24s} acc {100 * r['test_accuracy_mean']:.2f}% (sd {100 * r['test_accuracy_sd']:.2f}) "
              f"recall {'-' if rc is None else f'{rc:.2f}'} terms {r['inference_terms_total']:>8d} "
              f"({r['inference_terms_total'] / ref:.2f}x H256) time {timing[arm][0] * 1000:7.1f} ms "
              f"({timing[arm][0] / tref:.2f}x H256)")
    print("A/A:", {k: round(v[0] * 1000, 1) for k, v in timing.items() if k.startswith("dense_H256")})


if __name__ == "__main__":
    main()
