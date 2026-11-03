"""CL-62: does low-precision accumulation explain the Q.ANT CPU-backend mismatch? (Tier 0)

Retrains the 18 CL-58 models with unchanged CL-58 code, evaluates simulator
variants with bfloat16 output rounding and low-precision accumulation, and
compares their disagreement counts with the archived Q.ANT counts
(CL-58 results_qant.json). See README.md and PREDICTIONS.md.

Run from the repository root:
    python -m experiments.cl62_qant_accumulation_simulator.run [--processes N]
"""
from __future__ import annotations

import experiments.cl58_qant_quantization_aware_training.run as cl58  # sets single-threaded BLAS first

import argparse
import json
import multiprocessing as mp
import statistics
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
QANT_RESULTS = cl58.HERE / "results_qant.json"
VARIANTS = ("sim_cl58", "out", "seq", "pair", "block32", "block256")
bf16 = cl58.bf16


def _round_output(kan, b):
    return bf16(bf16(kan) + bf16(b))


def _terms(model, X):
    """bf16(bf16(cos) * bf16(A)), shape (B, C, D * grid), pixel-major."""
    _, act, A = model._act(X, quant=True)
    t = bf16(act * A[None])
    return t.reshape(t.shape[0], t.shape[1], -1)


def accumulate(t, mode):
    """Sum terms t (B, C, n) along the last axis with the given precision scheme."""
    n = t.shape[-1]
    if mode == "seq":
        s = np.zeros(t.shape[:2], np.float32)
        for j in range(n):
            s = bf16(s + t[:, :, j])
        return s
    if mode == "pair":
        size = 1 << (n - 1).bit_length()
        t = np.concatenate([t, np.zeros(t.shape[:2] + (size - n,), np.float32)], axis=2)
        while t.shape[-1] > 1:
            t = bf16(t[:, :, 0::2] + t[:, :, 1::2])
        return t[:, :, 0]
    if mode.startswith("block"):
        blk = int(mode[5:])
        pad = (-n) % blk
        t = np.concatenate([t, np.zeros(t.shape[:2] + (pad,), np.float32)], axis=2)
        partial = bf16(t.reshape(t.shape[0], t.shape[1], -1, blk).sum(-1, dtype=np.float32))
        return partial.sum(-1, dtype=np.float32)
    raise ValueError(mode)


def variant_logits(model, X, variant):
    if variant == "sim_cl58":
        return model.logits(X, quant=True)
    if variant == "out":
        _, act, A = model._act(X, quant=True)
        return _round_output(np.einsum("bcik,cik->bc", act, A), model.b)
    return _round_output(accumulate(_terms(model, X), variant), model.b)


def predict(model, X, variant, batch=100):
    return np.concatenate([variant_logits(model, X[s:s + batch], variant).argmax(1)
                           for s in range(0, len(X), batch)])


def train(seed, grid, arm, x_tr, y_tr):
    """Identical to CL-58 run_one's training loop."""
    model = cl58.QFourier(seed, grid)
    rng = np.random.default_rng(seed + 1)
    noise_rng = np.random.default_rng(seed + 2)
    for _ in range(cl58.EPOCHS):
        order = rng.permutation(len(x_tr))
        for s in range(0, len(order), cl58.BATCH):
            idx = order[s:s + cl58.BATCH]
            model.train_batch(x_tr[idx], y_tr[idx], arm, noise_rng)
    return model


def run_job(args):
    seed, grid, arm, data_dir = args
    x_tr, y_tr, x_te, y_te = cl58.load_fashion_mnist(Path(data_dir))
    model = train(seed, grid, arm, x_tr, y_tr)
    ref = model.predict(x_te)
    row = {"seed": seed, "grid": grid, "arm": arm, "reference_accuracy": float((ref == y_te).mean())}
    for v in VARIANTS:
        p = predict(model, x_te, v)
        row[f"{v}_accuracy"] = float((p == y_te).mean())
        row[f"{v}_disagreements"] = int((p != ref).sum())
    return row


def pearson(a, b):
    return float(np.corrcoef(np.asarray(a, float), np.asarray(b, float))[0, 1])


def run(data_dir: Path, processes=4):
    qant = {(r["seed"], r["grid"], r["arm"]): r for r in json.loads(QANT_RESULTS.read_text())["rows"]}
    cl58.load_fashion_mnist(data_dir)
    jobs = [(s, g, a, str(data_dir)) for g in cl58.GRIDS for a in cl58.ARMS for s in cl58.SEEDS]
    with mp.get_context("fork").Pool(processes) as pool:
        rows = pool.map(run_job, jobs)
    for r in rows:
        q = qant[(r["seed"], r["grid"], r["arm"])]
        r["archived_ref_vs_sim"] = q["disagreements_ref_vs_sim"]
        r["archived_reference_accuracy"] = q["reference_accuracy"]
        r["qant_disagreements"] = q["disagreements_ref_vs_qant"]
        r["qant_accuracy"] = q["qant_accuracy"]
    reproduced = all(r["sim_cl58_disagreements"] == r["archived_ref_vs_sim"]
                     and r["reference_accuracy"] == r["archived_reference_accuracy"] for r in rows)
    summary = {}
    for g in cl58.GRIDS:
        rs = [r for r in rows if r["grid"] == g]
        qm = statistics.fmean(r["qant_disagreements"] for r in rs)
        summary[f"grid{g}"] = {"qant_mean_disagreements": qm, "qant_mean_accuracy": statistics.fmean(r["qant_accuracy"] for r in rs)}
        for v in VARIANTS:
            m = statistics.fmean(r[f"{v}_disagreements"] for r in rs)
            summary[f"grid{g}"][v] = {"mean_disagreements": m, "ratio_to_qant": m / qm,
                                      "mean_accuracy": statistics.fmean(r[f"{v}_accuracy"] for r in rs),
                                      "pearson_within_grid": pearson([r[f"{v}_disagreements"] for r in rs],
                                                                     [r["qant_disagreements"] for r in rs])}
    corr_all = {v: pearson([r[f"{v}_disagreements"] for r in rows], [r["qant_disagreements"] for r in rows]) for v in VARIANTS}
    return {"experiment": "CL-62", "dataset": "FashionMNIST", "variants": list(VARIANTS),
            "term_order": "pixel-major (i, then k) within each logit",
            "qant_source": "experiments/cl58_qant_quantization_aware_training/results_qant.json",
            "reproduced_cl58_exactly": reproduced, "numpy": np.__version__,
            "rows": rows, "summary": summary, "pearson_all_18_runs": corr_all}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=cl58.HERE / "data")
    ap.add_argument("--processes", type=int, default=4)
    args = ap.parse_args()
    res = run(args.data_dir, args.processes)
    (HERE / "results.json").write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")
    print("reproduced CL-58 exactly:", res["reproduced_cl58_exactly"])
    for g, s in res["summary"].items():
        print(f"{g}: Q.ANT {s['qant_mean_disagreements']:.1f} dis, acc {s['qant_mean_accuracy']:.4f}")
        for v in VARIANTS:
            print(f"  {v:9s} {s[v]['mean_disagreements']:8.1f} dis  ratio {s[v]['ratio_to_qant']:.2f}"
                  f"  acc {s[v]['mean_accuracy']:.4f}  r_within {s[v]['pearson_within_grid']:+.2f}")
    print("pearson over 18 runs:", {v: round(c, 2) for v, c in res["pearson_all_18_runs"].items()})


if __name__ == "__main__":
    main()
