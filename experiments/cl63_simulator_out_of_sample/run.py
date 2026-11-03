"""CL-63: out-of-sample test of the CL-62 Q.ANT accumulation simulator (Tier 0).

A. Exp021: regenerate the synthetic KAN layers exactly and compare simulated MAE
   against the archived Q.ANT MAE per condition.
B. Exp022: grouped accumulation, simulated on the CL-58 exact_cosine grid-4 models,
   compared with the archived Q.ANT disagreement counts.
See README.md and PREDICTIONS.md.

Run from the repository root:
    python -m experiments.cl63_simulator_out_of_sample.run
"""
from __future__ import annotations

import experiments.cl62_qant_accumulation_simulator.run as cl62  # sets single-threaded BLAS first

import argparse
import json
import multiprocessing as mp
import statistics
from pathlib import Path

import numpy as np

cl58 = cl62.cl58
bf16 = cl58.bf16
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXP021 = ROOT / "results" / "exp021_qant_accumulation_scaling_law.json"
EXP022 = ROOT / "results" / "exp022_qant_grouped_accumulation_architecture.json"
VARIANTS_A = ("f32acc", "seq", "seq_kmajor", "pair", "block32")
GROUPS_B = (1, 4, 16)


# ------------------------------------------------------------------ A. Exp021

def exp021_inputs(seed, ni, nk, n=256, out=10):
    """Identical to experiments/021_qant_accumulation_scaling_law/run.py."""
    rng = np.random.default_rng(seed * 100000 + ni * 100 + nk)
    x = rng.uniform(-1, 1, size=(n, ni)).astype(np.float32)
    p = rng.uniform(-np.pi / 2, np.pi / 2, size=(out, ni, nk)).astype(np.float32)
    a = rng.normal(0, 1 / np.sqrt(ni * nk), size=(out, ni, nk)).astype(np.float32)
    k = np.arange(1, nk + 1, dtype=np.float32)
    exact = np.sum(a[None] * np.cos(x[:, None, :, None] * k[None, None, None, :] + p[None]), axis=(2, 3)).astype(np.float32)
    return x, p, a, k, exact


def kan_terms(x, p, a, k):
    """bf16 operands -> bf16 cos -> bf16 products, shape (B, C, ni, nk)."""
    z = bf16(x)[:, None, :, None] * k[None, None, None, :] + bf16(p)[None]
    act = bf16(np.cos(z))
    return act, bf16(act * bf16(a)[None])


def kan_sim(x, p, a, k, variant):
    act, t = kan_terms(x, p, a, k)
    B, C = t.shape[:2]
    if variant == "f32acc":
        return bf16(np.einsum("bcik,cik->bc", act, bf16(a)))
    if variant == "seq_kmajor":
        return bf16(cl62.accumulate(np.ascontiguousarray(t.transpose(0, 1, 3, 2)).reshape(B, C, -1), "seq"))
    return bf16(cl62.accumulate(t.reshape(B, C, -1), variant))


def run_a():
    arch = json.loads(EXP021.read_text())
    rows = []
    for r in arch["results"]:
        x, p, a, k, exact = exp021_inputs(r["seed"], r["input_channels"], r["components"])
        row = {"seed": r["seed"], "input_channels": r["input_channels"], "components": r["components"],
               "terms": r["accumulated_terms"], "qant_mae": r["qant_mae_vs_exact_cosine"],
               "qant_rmse": r["qant_rmse_vs_exact_cosine"]}
        for v in VARIANTS_A:
            err = kan_sim(x, p, a, k, v) - exact
            row[f"{v}_mae"] = float(np.abs(err).mean())
            row[f"{v}_rmse"] = float(np.sqrt(np.mean(err * err)))
        rows.append(row)
    conds = sorted({(r["input_channels"], r["components"]) for r in rows})
    per_cond = []
    for ni, nk in conds:
        rs = [r for r in rows if r["input_channels"] == ni and r["components"] == nk]
        c = {"input_channels": ni, "components": nk, "terms": ni * nk,
             "qant_mae": statistics.fmean(r["qant_mae"] for r in rs)}
        for v in VARIANTS_A:
            c[f"{v}_mae"] = statistics.fmean(r[f"{v}_mae"] for r in rs)
            c[f"{v}_ratio"] = c[f"{v}_mae"] / c["qant_mae"]
        per_cond.append(c)
    lt = np.log([r["terms"] for r in rows])
    fits = {"qant": float(np.polyfit(lt, np.log([r["qant_mae"] for r in rows]), 1)[0])}
    for v in VARIANTS_A:
        fits[v] = float(np.polyfit(lt, np.log([r[f"{v}_mae"] for r in rows]), 1)[0])
    agg = {v: {"exponent": fits[v],
               "share_conditions_within_15pct": float(np.mean([abs(c[f"{v}_ratio"] - 1) <= 0.15 for c in per_cond])),
               "median_ratio": float(np.median([c[f"{v}_ratio"] for c in per_cond])),
               "pearson_log_mae_108": float(np.corrcoef(np.log([r[f"{v}_mae"] for r in rows]),
                                                        np.log([r["qant_mae"] for r in rows]))[0, 1])}
           for v in VARIANTS_A}
    return {"qant_exponent": fits["qant"], "archived_qant_exponent": arch["scaling_analysis"]["power_law_exponent"],
            "variants": agg, "per_condition": per_cond, "rows": rows}


# ------------------------------------------------------------------ B. Exp022

def grouped_logits(model, X, groups, variant):
    """Exp022: one Q.ANT call per contiguous pixel group, float32 sum of bf16 outputs, bf16 bias."""
    total = np.zeros((len(X), cl58.C), np.float32)
    for g in range(groups):
        lo, hi = g * cl58.D // groups, (g + 1) * cl58.D // groups
        out = kan_sim(X[:, lo:hi], model.phi[:, lo:hi], model.A[:, lo:hi], model.k, variant)
        total = total + out
    return bf16(bf16(total) + bf16(model.b))


def run_b_job(args):
    seed, data_dir = args
    x_tr, y_tr, x_te, y_te = cl58.load_fashion_mnist(Path(data_dir))
    model = cl62.train(seed, 4, "exact_cosine", x_tr, y_tr)
    ref = model.predict(x_te)
    row = {"seed": seed, "reference_accuracy": float((ref == y_te).mean())}
    for v in ("seq", "f32acc"):
        for g in GROUPS_B:
            pred = np.concatenate([grouped_logits(model, x_te[s:s + 100], g, v).argmax(1) for s in range(0, len(x_te), 100)])
            row[f"{v}_g{g}_disagreements"] = int((pred != ref).sum())
            row[f"{v}_g{g}_accuracy"] = float((pred == y_te).mean())
    return row


def run_b(data_dir, processes):
    arch = json.loads(EXP022.read_text())["results"]
    qant = {name: statistics.fmean(r["prediction_disagreements"] for r in arch if r["candidate_id"] == name)
            for name in ("monolithic_g4", "grouped_g4_4", "grouped_g4_16")}
    cl58.load_fashion_mnist(data_dir)
    with mp.get_context("fork").Pool(processes) as pool:
        rows = pool.map(run_b_job, [(s, str(data_dir)) for s in cl58.SEEDS])
    names = {1: "monolithic_g4", 4: "grouped_g4_4", 16: "grouped_g4_16"}
    summary = {}
    for g in GROUPS_B:
        q = qant[names[g]]
        summary[f"groups_{g}"] = {"exp022_qant_mean_disagreements": q}
        for v in ("seq", "f32acc"):
            m = statistics.fmean(r[f"{v}_g{g}_disagreements"] for r in rows)
            summary[f"groups_{g}"][v] = {"mean_disagreements": m, "ratio_to_qant": m / q}
    return {"models": "CL-58 exact_cosine grid-4 models (NumPy), not the Exp022 torch models",
            "summary": summary, "rows": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=cl58.HERE / "data")
    ap.add_argument("--processes", type=int, default=3)
    args = ap.parse_args()
    a = run_a()
    b = run_b(args.data_dir, args.processes)
    res = {"experiment": "CL-63", "numpy": np.__version__, "exp021": a, "exp022": b}
    (HERE / "results.json").write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")
    print(f"A. Exp021: Q.ANT exponent {a['qant_exponent']:.3f} (archived {a['archived_qant_exponent']:.3f})")
    for v, s in a["variants"].items():
        print(f"  {v:11s} exponent {s['exponent']:.3f}  within15% {s['share_conditions_within_15pct']:.2f}"
              f"  median ratio {s['median_ratio']:.2f}  r(log) {s['pearson_log_mae_108']:.3f}")
    for c in a["per_condition"]:
        if c["terms"] in (1, 784 * 8) or (c["input_channels"], c["components"]) in ((64, 4), (784, 1)):
            print("   ", c["input_channels"], c["components"], f"qant {c['qant_mae']:.5f}",
                  " ".join(f"{v} {c[v + '_ratio']:.2f}" for v in VARIANTS_A))
    print("B. Exp022:")
    for g, s in b["summary"].items():
        print(f"  {g:10s} Q.ANT {s['exp022_qant_mean_disagreements']:.1f}  seq {s['seq']['mean_disagreements']:.1f}"
              f" ({s['seq']['ratio_to_qant']:.2f})  f32acc {s['f32acc']['mean_disagreements']:.1f}")


if __name__ == "__main__":
    main()
