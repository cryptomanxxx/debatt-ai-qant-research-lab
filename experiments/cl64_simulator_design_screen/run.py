"""CL-64: screening Q.ANT Fourier/KAN designs with the accumulation simulator (Tier 0).

1. Grouped monolithic: CL-58 exact_cosine grid-4 models, logit sum split into g Q.ANT calls.
2. Hierarchical (Exp027 designs): g blocks of 784/g pixels -> w features, head g*w -> 10,
   reimplemented and trained in NumPy, each stage simulated as Q.ANT calls.
Every Q.ANT call is simulated with CL-62's sequential-bfloat16 accumulation. See README.md
and PREDICTIONS.md.

Run from the repository root:
    python -m experiments.cl64_simulator_design_screen.run [--processes N]
"""
from __future__ import annotations

import experiments.cl63_simulator_out_of_sample.run as cl63  # sets single-threaded BLAS first

import argparse
import json
import multiprocessing as mp
import statistics
from pathlib import Path

import numpy as np

cl62, cl58 = cl63.cl62, cl63.cl58
bf16 = cl58.bf16
HERE = Path(__file__).resolve().parent
EXP027 = cl63.ROOT / "results" / "exp027_qant_factorial_accumulation_map.json"
SEEDS = cl58.SEEDS
G = 4
K = np.arange(1, G + 1, dtype=np.float32)
GROUP_COUNTS = (1, 2, 4, 8, 16, 49, 98, 196, 784)
HIER = [(g, w) for g in (8, 16, 28) for w in (2, 8)]
EPOCHS, BATCH, LR = 3, 128, 1e-3


# ------------------------------------------------------------------ simulated Q.ANT call

def qant_call(X, phi, A, b, mode):
    """One calc_kan_layer_fprop + add_bias_fprop. X (B, ni), phi/A (no, ni, G), b (no)."""
    if mode == "exact":
        z = X[:, None, :, None] * K + phi[None]
        return np.einsum("bcik,cik->bc", np.cos(z), A) + b
    out = cl63.kan_sim(X, phi, A, K, "seq" if mode == "seq" else "f32acc")
    return bf16(bf16(out) + bf16(b))


# ------------------------------------------------------------------ hierarchical model

def init_layer(rng, ni, no, lead=()):
    std = (2 / ((ni + no) * G)) ** 0.5
    return {"A": (rng.standard_normal(lead + (no, ni, G)) * std).astype(np.float32),
            "phi": rng.uniform(-np.pi / 2, np.pi / 2, lead + (no, ni, G)).astype(np.float32),
            "b": np.zeros(lead + (no,), np.float32)}


class Hier:
    """Exp027 H(g, w): blocks share nothing; stage-1 parameters are stacked over blocks."""

    def __init__(self, seed, g, w):
        rng = np.random.default_rng(seed)
        self.g, self.w, self.chunk = g, w, cl58.D // g
        self.p = {f"s1_{k}": v for k, v in init_layer(rng, self.chunk, w, (g,)).items()}
        self.p.update({f"s2_{k}": v for k, v in init_layer(rng, g * w, cl58.C).items()})
        self.m = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.t = 0

    def nparams(self):
        return int(sum(v.size for v in self.p.values()))

    def stage1(self, X):
        Xb = X[:, : self.g * self.chunk].reshape(len(X), self.g, self.chunk)
        z = Xb[:, :, None, :, None] * K + self.p["s1_phi"][None]          # (B, g, w, chunk, G)
        h = np.einsum("bgwik,gwik->bgw", np.cos(z), self.p["s1_A"]) + self.p["s1_b"]
        return Xb, z, h.reshape(len(X), -1)

    def stage2(self, H):
        z = H[:, None, :, None] * K + self.p["s2_phi"][None]              # (B, 10, g*w, G)
        return z, np.einsum("bcik,cik->bc", np.cos(z), self.p["s2_A"]) + self.p["s2_b"]

    def logits(self, X):
        return self.stage2(self.stage1(X)[2])[1]

    def train_batch(self, X, y):
        grads = self.grads(X, y)
        self.t += 1
        for k, gr in grads.items():
            self.m[k] = 0.9 * self.m[k] + 0.1 * gr
            self.v[k] = 0.999 * self.v[k] + 0.001 * gr * gr
            mh = self.m[k] / (1 - 0.9 ** self.t)
            vh = self.v[k] / (1 - 0.999 ** self.t)
            self.p[k] -= (LR * mh / (np.sqrt(vh) + 1e-8)).astype(np.float32)

    def grads(self, X, y):
        Xb, z1, H = self.stage1(X)
        z2, logits = self.stage2(H)
        logits = logits - logits.max(1, keepdims=True)
        p = np.exp(logits)
        p /= p.sum(1, keepdims=True)
        p[np.arange(len(y)), y] -= 1
        g2 = (p / len(y)).astype(np.float32)
        s2 = np.sin(z2)
        grads = {"s2_A": np.einsum("bc,bcik->cik", g2, np.cos(z2)),
                 "s2_phi": -self.p["s2_A"] * np.einsum("bc,bcik->cik", g2, s2),
                 "s2_b": g2.sum(0)}
        dH = -np.einsum("bc,cik,bcik->bi", g2, self.p["s2_A"] * K, s2)
        dh = dH.reshape(len(X), self.g, self.w)
        s1 = np.sin(z1)
        grads["s1_A"] = np.einsum("bgw,bgwik->gwik", dh, np.cos(z1))
        grads["s1_phi"] = -self.p["s1_A"] * np.einsum("bgw,bgwik->gwik", dh, s1)
        grads["s1_b"] = dh.sum(0)
        return grads

    def sim_logits(self, X, mode1, mode2):
        """Each stage-1 block and the head as separate simulated Q.ANT calls."""
        hs = [qant_call(X[:, i * self.chunk:(i + 1) * self.chunk], self.p["s1_phi"][i], self.p["s1_A"][i],
                        self.p["s1_b"][i], mode1) for i in range(self.g)]
        H = np.concatenate(hs, axis=1).astype(np.float32)
        return qant_call(H, self.p["s2_phi"], self.p["s2_A"], self.p["s2_b"], mode2)


def predict(fn, X, batch=200):
    return np.concatenate([fn(X[s:s + batch]).argmax(1) for s in range(0, len(X), batch)])


# ------------------------------------------------------------------ jobs

def hier_job(args):
    seed, g, w, data_dir = args
    x_tr, y_tr, x_te, y_te = cl58.load_fashion_mnist(Path(data_dir))
    model = Hier(seed, g, w)
    rng = np.random.default_rng(seed + 1)
    for _ in range(EPOCHS):
        order = rng.permutation(len(x_tr))
        for s in range(0, len(order), BATCH):
            idx = order[s:s + BATCH]
            model.train_batch(x_tr[idx], y_tr[idx])
    ref = predict(model.logits, x_te)
    row = {"design": f"hierarchical_{g}x{w}_g4", "kind": "hierarchical", "seed": seed, "groups": g, "width": w,
           "params": model.nparams(), "qant_calls": g + 1, "stage1_terms": model.chunk * G, "stage2_terms": g * w * G,
           "reference_accuracy": float((ref == y_te).mean())}
    for name, m1, m2 in (("seq", "seq", "seq"), ("f32acc", "f32acc", "f32acc"),
                         ("seq_stage1_only", "seq", "exact"), ("seq_stage2_only", "exact", "seq")):
        pred = predict(lambda X: model.sim_logits(X, m1, m2), x_te)
        row[f"{name}_disagreements"] = int((pred != ref).sum())
        row[f"{name}_accuracy"] = float((pred == y_te).mean())
    return row


def grouped_job(args):
    seed, data_dir = args
    x_tr, y_tr, x_te, y_te = cl58.load_fashion_mnist(Path(data_dir))
    model = cl62.train(seed, G, "exact_cosine", x_tr, y_tr)
    ref = model.predict(x_te)
    rows = []
    for g in GROUP_COUNTS:
        row = {"design": f"grouped_monolithic_g{g}", "kind": "grouped", "seed": seed, "groups": g,
               "params": int(model.A.size + model.phi.size + model.b.size), "qant_calls": g,
               "stage1_terms": (cl58.D // g) * G, "reference_accuracy": float((ref == y_te).mean())}
        for v in ("seq", "f32acc"):
            pred = predict(lambda X: cl63.grouped_logits(model, X, g, v, bias=True), x_te)
            row[f"{v}_disagreements"] = int((pred != ref).sum())
            row[f"{v}_accuracy"] = float((pred == y_te).mean())
        rows.append(row)
    return rows


def summarize(rows):
    out = {}
    for d in dict.fromkeys(r["design"] for r in rows):
        rs = [r for r in rows if r["design"] == d]
        keys = [k for k in rs[0] if isinstance(rs[0][k], (int, float)) and k != "seed"]
        out[d] = {k: statistics.fmean(r[k] for r in rs) for k in keys}
    return out


def run(data_dir, processes):
    cl58.load_fashion_mnist(data_dir)
    jobs_h = [(s, g, w, str(data_dir)) for (g, w) in HIER for s in SEEDS]
    with mp.get_context("fork").Pool(processes) as pool:
        grouped = pool.map_async(grouped_job, [(s, str(data_dir)) for s in SEEDS])
        hier = pool.map(hier_job, jobs_h)
        grouped = [r for rs in grouped.get() for r in rs]
    arch = json.loads(EXP027.read_text())["summary"]
    summary = summarize(grouped + hier)
    for d, s in summary.items():
        if d in arch:
            s["exp027_reference_accuracy"] = arch[d]["mean_reference_accuracy"]
            s["exp027_qant_accuracy"] = arch[d]["mean_qant_accuracy"]
            s["exp027_qant_disagreements"] = arch[d]["mean_prediction_disagreements"]
            s["seq_ratio_to_exp027"] = s["seq_disagreements"] / arch[d]["mean_prediction_disagreements"]
    gs = [summary[f"grouped_monolithic_g{g}"]["seq_disagreements"] for g in GROUP_COUNTS if g <= 49]
    slope = float(np.polyfit(np.log([g for g in GROUP_COUNTS if g <= 49]), np.log(gs), 1)[0])
    return {"experiment": "CL-64", "dataset": "FashionMNIST", "grid": G, "epochs": EPOCHS, "seeds": list(SEEDS),
            "simulator": "CL-62 seq: bf16 operands/cos/products, bf16 running sum, bf16 bias per Q.ANT call",
            "grouped_note": "group outputs summed in float32, bf16 bias added once (unlike Exp022, which omitted it)",
            "hierarchical_note": "NumPy reimplementation of Exp027 designs, not the archived torch models",
            "grouped_loglog_slope_g1_to_49": slope, "numpy": np.__version__,
            "summary": summary, "rows": grouped + hier}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=cl58.HERE / "data")
    ap.add_argument("--processes", type=int, default=4)
    args = ap.parse_args()
    res = run(args.data_dir, args.processes)
    (HERE / "results.json").write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")
    print(f"grouped log-log slope (g 1-49): {res['grouped_loglog_slope_g1_to_49']:.2f}")
    for d, s in res["summary"].items():
        line = (f"{d:26s} calls {s['qant_calls']:5.0f} ref {s['reference_accuracy']:.4f} sim-acc {s['seq_accuracy']:.4f}"
                f" dis seq {s['seq_disagreements']:6.1f} f32 {s['f32acc_disagreements']:5.1f}")
        if "exp027_qant_disagreements" in s:
            line += (f" | Exp027 ref {s['exp027_reference_accuracy']:.4f} dis {s['exp027_qant_disagreements']:.1f}"
                     f" ratio {s['seq_ratio_to_exp027']:.2f} | s1-only {s['seq_stage1_only_disagreements']:.1f}"
                     f" s2-only {s['seq_stage2_only_disagreements']:.1f}")
        print(line)


if __name__ == "__main__":
    main()
