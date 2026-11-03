"""CL-65: hierarchical Fourier/KAN with the head split into several Q.ANT calls (Tier 0).

Trains the CL-64 28x8 and 16x8 hierarchical designs (same code and seeds) and evaluates
them with the head sum split into h simulated Q.ANT calls (float32 sum of the group
outputs, bf16 bias once). See README.md and PREDICTIONS.md.

Run from the repository root:
    python -m experiments.cl65_hierarchical_grouped_head.run [--processes N]
"""
from __future__ import annotations

import experiments.cl64_simulator_design_screen.run as cl64  # sets single-threaded BLAS first

import argparse
import json
import multiprocessing as mp
import statistics
from pathlib import Path

import numpy as np

cl58, cl63 = cl64.cl58, cl64.cl63
bf16 = cl58.bf16
HERE = Path(__file__).resolve().parent
DESIGNS = {(28, 8): (1, 2, 4, 8, 16, 28), (16, 8): (1, 2, 4, 8, 16)}


def grouped_head(model, H, h, mode):
    """Head as h simulated Q.ANT calls over contiguous feature groups; bf16 bias once."""
    A, phi, b = model.p["s2_A"], model.p["s2_phi"], model.p["s2_b"]
    if mode == "exact":
        return cl64.qant_call(H, phi, A, b, "exact")
    n = H.shape[1]
    total = np.zeros((len(H), cl58.C), np.float32)
    for i in range(h):
        lo, hi = i * n // h, (i + 1) * n // h
        total = total + cl63.kan_sim(H[:, lo:hi], phi[:, lo:hi], A[:, lo:hi], cl64.K, "seq")
    return bf16(bf16(total) + bf16(b))


def stage1(model, X, mode):
    hs = [cl64.qant_call(X[:, i * model.chunk:(i + 1) * model.chunk], model.p["s1_phi"][i], model.p["s1_A"][i],
                         model.p["s1_b"][i], mode) for i in range(model.g)]
    return np.concatenate(hs, axis=1).astype(np.float32)


def job(args):
    seed, g, w, data_dir = args
    x_tr, y_tr, x_te, y_te = cl58.load_fashion_mnist(Path(data_dir))
    model = cl64.Hier(seed, g, w)
    rng = np.random.default_rng(seed + 1)
    for _ in range(cl64.EPOCHS):
        order = rng.permutation(len(x_tr))
        for s in range(0, len(order), cl64.BATCH):
            idx = order[s:s + cl64.BATCH]
            model.train_batch(x_tr[idx], y_tr[idx])
    ref = cl64.predict(model.logits, x_te)
    rows = []
    floor = cl64.predict(lambda X: grouped_head(model, stage1(model, X, "seq"), 1, "exact"), x_te)
    for h in DESIGNS[(g, w)]:
        full = cl64.predict(lambda X: grouped_head(model, stage1(model, X, "seq"), h, "seq"), x_te)
        head_only = cl64.predict(lambda X: grouped_head(model, stage1(model, X, "exact"), h, "seq"), x_te)
        rows.append({"design": f"hierarchical_{g}x{w}_g4", "seed": seed, "head_calls": h, "qant_calls": g + h,
                     "head_terms_per_call": (g * w // h) * cl64.G, "reference_accuracy": float((ref == y_te).mean()),
                     "sim_accuracy": float((full == y_te).mean()), "sim_disagreements": int((full != ref).sum()),
                     "head_only_disagreements": int((head_only != ref).sum()),
                     "stage1_floor_disagreements": int((floor != ref).sum())})
    return rows


def run(data_dir, processes):
    cl58.load_fashion_mnist(data_dir)
    jobs = [(s, g, w, str(data_dir)) for (g, w) in DESIGNS for s in cl64.SEEDS]
    with mp.get_context("fork").Pool(processes) as pool:
        rows = [r for rs in pool.map(job, jobs) for r in rs]
    summary = {}
    for (g, w), hs in DESIGNS.items():
        d = f"hierarchical_{g}x{w}_g4"
        for h in hs:
            rs = [r for r in rows if r["design"] == d and r["head_calls"] == h]
            keys = [k for k in rs[0] if isinstance(rs[0][k], (int, float)) and k != "seed"]
            summary[f"{d}_head{h}"] = {k: statistics.fmean(r[k] for r in rs) for k in keys}
    return {"experiment": "CL-65", "dataset": "FashionMNIST", "seeds": list(cl64.SEEDS), "numpy": np.__version__,
            "simulator": "CL-62 seq per Q.ANT call; head groups summed in float32, bf16 bias once",
            "summary": summary, "rows": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=cl58.HERE / "data")
    ap.add_argument("--processes", type=int, default=4)
    args = ap.parse_args()
    res = run(args.data_dir, args.processes)
    (HERE / "results.json").write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")
    for k, s in res["summary"].items():
        print(f"{k:30s} calls {s['qant_calls']:4.0f} terms/head-call {s['head_terms_per_call']:5.0f} ref {s['reference_accuracy']:.4f}"
              f" sim-acc {s['sim_accuracy']:.4f} dis {s['sim_disagreements']:6.1f} head-only {s['head_only_disagreements']:6.1f}"
              f" floor {s['stage1_floor_disagreements']:5.1f}")


if __name__ == "__main__":
    main()
