"""CL-69: a systematic model zoo for Q.ANT numerical fidelity (Tier 0, phase C).

Models are stacks of Linear / KAN layers with ReLU or sigmoid, trained in float32 NumPy,
then evaluated along a simulated Q.ANT CPU-backend path:
  calc_kan_layer_fprop -> bf16 operands/cos/products, sequential bf16 sum (CL-62..66)
  linear_fprop         -> bf16 inputs/weights/outputs, float32 accumulation (CL-68)
  add_bias_fprop       -> bf16;  relu exact on bf16;  sigmoid float32 with bf16 output
The CL-65 hierarchical design is included via its own code. See PREDICTIONS.md.

Run from the repository root:
    python -m experiments.cl69_qant_model_zoo.run [--processes N]
"""
from __future__ import annotations

import experiments.cl65_hierarchical_grouped_head.run as cl65  # sets single-threaded BLAS first
import experiments.cl67_accumulation_aware_training.run as cl67

import argparse
import json
import multiprocessing as mp
import statistics
from pathlib import Path

import numpy as np

cl64, cl58 = cl65.cl64, cl65.cl58
bf16 = cl58.bf16
HERE = Path(__file__).resolve().parent
SEEDS = (11, 22, 33)
EPOCHS, BATCH, LR = 3, 128, 1e-3

# name: (layer spec list, eval variants {label: calls-per-KAN-layer}); "act" applies after every hidden layer
ZOO = {
    "logreg":         ([("lin", 10)], "relu"),
    "mlp_h64":        ([("lin", 64), ("lin", 10)], "relu"),
    "mlp_h256":       ([("lin", 256), ("lin", 10)], "relu"),
    "mlp_h1024":      ([("lin", 1024), ("lin", 10)], "relu"),
    "mlp_h256x2":     ([("lin", 256), ("lin", 256), ("lin", 10)], "relu"),
    "mlp_h256x3":     ([("lin", 256), ("lin", 256), ("lin", 256), ("lin", 10)], "relu"),
    "mlp_sig_h256":   ([("lin", 256), ("lin", 10)], "sigmoid"),
    "kan_g1":         ([("kan", 10, 1)], None),
    "kan_g2":         ([("kan", 10, 2)], None),
    "kan_g4":         ([("kan", 10, 4)], None),
    "kan_g8":         ([("kan", 10, 8)], None),
    "lin128_kanhead": ([("lin", 128), ("kan", 10, 4)], "relu"),
    "kan32_linhead":  ([("kan", 32, 4), ("lin", 10)], None),
}
EXTRA = {"kan_g4_split16": ("kan_g4", 16)}   # same trained model, evaluated with 16 calls per logit
HIER = "hier_28x8_head8"


# ------------------------------------------------------------------ layers

class Linear:
    def __init__(self, rng, ni, no):
        bound = 1 / np.sqrt(ni)
        self.p = {"W": rng.uniform(-bound, bound, (no, ni)).astype(np.float32),
                  "b": rng.uniform(-bound, bound, no).astype(np.float32)}
        self.terms = 0  # no sequential bf16 sum (float32 accumulation)
        self.macs = ni * no

    def forward(self, X):
        return X @ self.p["W"].T + self.p["b"], X

    def backward(self, g, cache):
        X = cache
        return g @ self.p["W"], {"W": g.T @ X, "b": g.sum(0)}

    def qant(self, X, calls=1):
        return bf16(bf16(bf16(X) @ bf16(self.p["W"]).T) + bf16(self.p["b"]))


class KAN:
    def __init__(self, rng, ni, no, G):
        std = (2 / ((ni + no) * G)) ** 0.5
        self.p = {"A": (rng.standard_normal((no, ni, G)) * std).astype(np.float32),
                  "phi": rng.uniform(-np.pi / 2, np.pi / 2, (no, ni, G)).astype(np.float32),
                  "b": np.zeros(no, np.float32)}
        self.k = np.arange(1, G + 1, dtype=np.float32)
        self.ni, self.G = ni, G
        self.terms = ni * G
        self.macs = ni * no * G

    def forward(self, X):
        z = X[:, None, :, None] * self.k + self.p["phi"][None]
        return np.einsum("bcik,cik->bc", np.cos(z), self.p["A"]) + self.p["b"], (X, z)

    def backward(self, g, cache):
        X, z = cache
        s = np.sin(z)
        grads = {"A": np.einsum("bc,bcik->cik", g, np.cos(z)),
                 "phi": -self.p["A"] * np.einsum("bc,bcik->cik", g, s), "b": g.sum(0)}
        dX = -np.einsum("bc,cik,bcik->bi", g, self.p["A"] * self.k, s)
        return dX, grads

    def qant(self, X, calls=1):
        """`calls` contiguous input groups, each one simulated calc_kan_layer_fprop; float32 sum; bf16 bias."""
        total = np.zeros((len(X), self.p["b"].size), np.float32)
        for c in range(calls):
            lo, hi = c * self.ni // calls, (c + 1) * self.ni // calls
            z = bf16(X[:, lo:hi])[:, None, :, None] * self.k + bf16(self.p["phi"][:, lo:hi])[None]
            t = bf16(bf16(np.cos(z)) * bf16(self.p["A"][:, lo:hi])[None]).reshape(len(X), self.p["b"].size, -1)
            total = total + bf16(cl67.seq_sum(t))
        return bf16(bf16(total) + bf16(self.p["b"]))


def act_fwd(name, z):
    return np.maximum(z, 0) if name == "relu" else 1 / (1 + np.exp(-z))


def act_bwd(name, a, g):
    return g * (a > 0) if name == "relu" else g * a * (1 - a)


def act_qant(name, z):
    return np.maximum(z, 0) if name == "relu" else bf16(1 / (1 + np.exp(-z)))


class Net:
    def __init__(self, seed, spec, act):
        rng = np.random.default_rng(seed)
        self.layers, ni = [], 784
        for s in spec:
            self.layers.append(Linear(rng, ni, s[1]) if s[0] == "lin" else KAN(rng, ni, s[1], s[2]))
            ni = s[1]
        self.act = act
        self.m = [{k: np.zeros_like(v) for k, v in L.p.items()} for L in self.layers]
        self.v = [{k: np.zeros_like(v) for k, v in L.p.items()} for L in self.layers]
        self.t = 0

    def params(self):
        return int(sum(v.size for L in self.layers for v in L.p.values()))

    def logits(self, X):
        z = X
        for i, L in enumerate(self.layers):
            z = L.forward(z)[0]
            if i < len(self.layers) - 1 and self.act:
                z = act_fwd(self.act, z)
        return z

    def train_batch(self, X, y):
        caches, acts, z = [], [], X
        for i, L in enumerate(self.layers):
            z, c = L.forward(z)
            caches.append(c)
            if i < len(self.layers) - 1 and self.act:
                z = act_fwd(self.act, z)
            acts.append(z)
        z = z - z.max(1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(1, keepdims=True)
        p[np.arange(len(y)), y] -= 1
        g = (p / len(y)).astype(np.float32)
        self.t += 1
        for i in range(len(self.layers) - 1, -1, -1):
            if i < len(self.layers) - 1 and self.act:
                g = act_bwd(self.act, acts[i], g)
            g, grads = self.layers[i].backward(g, caches[i])
            for k, gr in grads.items():
                self.m[i][k] = 0.9 * self.m[i][k] + 0.1 * gr
                self.v[i][k] = 0.999 * self.v[i][k] + 0.001 * gr * gr
                mh = self.m[i][k] / (1 - 0.9 ** self.t)
                vh = self.v[i][k] / (1 - 0.999 ** self.t)
                self.layers[i].p[k] -= (LR * mh / (np.sqrt(vh) + 1e-8)).astype(np.float32)

    def qant_logits(self, X, calls=1):
        z = bf16(X)
        for i, L in enumerate(self.layers):
            z = L.qant(z, calls if isinstance(L, KAN) else 1)
            if i < len(self.layers) - 1 and self.act:
                z = act_qant(self.act, z)
        return z

    def longest_seq_sum(self, calls=1):
        return max([L.terms // (calls if isinstance(L, KAN) else 1) for L in self.layers] + [0])

    def qant_calls(self, calls=1):
        return sum(calls if isinstance(L, KAN) else 1 for L in self.layers)

    def macs(self):
        return int(sum(L.macs for L in self.layers))


def predict(fn, X, batch=250):
    return np.concatenate([fn(X[s:s + batch]).argmax(1) for s in range(0, len(X), batch)])


def train(model, seed, x_tr, y_tr):
    rng = np.random.default_rng(seed + 1)
    for _ in range(EPOCHS):
        order = rng.permutation(len(x_tr))
        for s in range(0, len(order), BATCH):
            idx = order[s:s + BATCH]
            model.train_batch(x_tr[idx], y_tr[idx])


def row(name, seed, ref, q, y, **info):
    return {"model": name, "seed": seed, "float_accuracy": float((ref == y).mean()),
            "qant_accuracy": float((q == y).mean()), "gap_pp": 100 * float((q == y).mean() - (ref == y).mean()),
            "disagreements": int((ref != q).sum()), **info}


def job(args):
    name, seed, data_dir = args
    x_tr, y_tr, x_te, y_te = cl58.load_fashion_mnist(Path(data_dir))
    if name == HIER:
        model = cl64.Hier(seed, 28, 8)
        train(model, seed, x_tr, y_tr)
        ref = cl64.predict(model.logits, x_te)
        q = cl64.predict(lambda X: cl65.grouped_head(model, cl65.stage1(model, X, "seq"), 8, "seq"), x_te)
        return [row(name, seed, ref, q, y_te, family="kan", params=model.nparams(), qant_calls=36,
                    longest_seq_sum=112, macs=784 * 8 * 4 + 224 * 10 * 4)]
    spec, act = ZOO[name]
    model = Net(seed, spec, act)
    train(model, seed, x_tr, y_tr)
    ref = predict(model.logits, x_te)
    fam = "linear" if all(s[0] == "lin" for s in spec) else ("kan" if all(s[0] == "kan" for s in spec) else "hybrid")
    out = [row(name, seed, ref, predict(model.qant_logits, x_te), y_te, family=fam, params=model.params(),
               qant_calls=model.qant_calls(), longest_seq_sum=model.longest_seq_sum(), macs=model.macs())]
    for extra, (base, calls) in EXTRA.items():
        if base == name:
            q = predict(lambda X: model.qant_logits(X, calls), x_te)
            out.append(row(extra, seed, ref, q, y_te, family=fam, params=model.params(),
                           qant_calls=model.qant_calls(calls), longest_seq_sum=model.longest_seq_sum(calls),
                           macs=model.macs()))
    return out


def average_ranks(x):
    """Ranks starting at 1; tied values get the mean of the ranks they span."""
    x = np.asarray(x, float)
    order = np.argsort(x, kind="stable")
    ranks = np.empty(len(x))
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and x[order[j + 1]] == x[order[i]]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def spearman(a, b):
    """Tie-corrected Spearman rank correlation (Pearson on average ranks)."""
    return float(np.corrcoef(average_ranks(a), average_ranks(b))[0, 1])


def run(data_dir, processes):
    cl58.load_fashion_mnist(data_dir)
    cl67.lib()
    names = list(ZOO) + [HIER]
    jobs = [(n, s, str(data_dir)) for n in sorted(names, key=lambda n: "kan" not in n) for s in SEEDS]
    with mp.get_context("fork").Pool(processes) as pool:
        rows = [r for rs in pool.map(job, jobs) for r in rs]
    order = list(ZOO) + list(EXTRA) + [HIER]
    summary = {}
    for n in order:
        rs = [r for r in rows if r["model"] == n]
        summary[n] = {k: (statistics.fmean(r[k] for r in rs) if isinstance(rs[0][k], (int, float)) else rs[0][k])
                      for k in rs[0] if k not in ("model", "seed")}
    kan = [n for n in order if summary[n]["family"] in ("kan", "hybrid")]
    rho = spearman([summary[n]["longest_seq_sum"] for n in kan], [summary[n]["disagreements"] for n in kan])
    return {"experiment": "CL-69", "dataset": "FashionMNIST", "seeds": list(SEEDS), "epochs": EPOCHS,
            "primitive_models": {"calc_kan_layer_fprop": "bf16 operands/cos/products, sequential bf16 sum",
                                 "linear_fprop": "bf16 in/weights/out, float32 accumulation",
                                 "add_bias_fprop": "bf16", "relu_fprop": "exact on bf16",
                                 "sigmoid_fprop": "float32 with bf16 output (assumption)",
                                 "final": "argmax of the bf16 logits (no softmax)"},
            "spearman_kan_longest_sum_vs_disagreements": rho, "numpy": np.__version__,
            "rows": rows, "summary": summary}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=cl58.HERE / "data")
    ap.add_argument("--processes", type=int, default=4)
    args = ap.parse_args()
    res = run(args.data_dir, args.processes)
    (HERE / "results.json").write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")
    print(f"{'model':16s} {'fam':6s} {'params':>7s} {'calls':>5s} {'maxsum':>6s} {'float':>7s} {'qant':>7s} {'gap pp':>7s} {'dis':>6s}")
    for n, s in res["summary"].items():
        print(f"{n:16s} {s['family']:6s} {s['params']:7.0f} {s['qant_calls']:5.0f} {s['longest_seq_sum']:6.0f}"
              f" {s['float_accuracy']:.4f} {s['qant_accuracy']:.4f} {s['gap_pp']:+7.2f} {s['disagreements']:6.1f}")
    print("Spearman (KAN-containing, longest seq sum vs disagreements):", round(res["spearman_kan_longest_sum_vs_disagreements"], 2))


if __name__ == "__main__":
    main()
