"""CL-70: fan-in budget of Q.ANT's linear and KAN primitives, plus the activations.

Synthetic, deterministic layers (seeded like Exp021) with fan-in from 1 to 65,536 terms.
  --backend simulator  local (Tier 0): simulated MAE for float32 and sequential-bf16
                       accumulation; needs gcc (C kernels from CL-67/CL-68).
  --backend qant       Q.ANT CPU backend (Tier 2, owner-approved; workflow
                       cl58-qant-qat.yml, experiment cl70): measured MAE of
                       linear_fprop and calc_kan_layer_fprop, and exact-match tests of
                       add_bias_fprop, relu_fprop, sigmoid_fprop and softmax_fprop.
The error is the mean absolute error against a float64 computation with the float32
parameters, divided by the mean absolute exact output (relative MAE). See PREDICTIONS.md.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SEEDS = (11, 22, 33)
N, OUT, G = 256, 10, 4
LINEAR_FANIN = (1, 4, 16, 64, 256, 1024, 4096, 16384, 65536)
KAN_INPUTS = (1, 4, 16, 64, 256, 1024, 4096, 16384)  # terms = inputs * G = 4 .. 65,536
KAN_CHUNK = 16


def bf16(a):
    a = np.ascontiguousarray(a, dtype=np.float32)
    u = a.view(np.uint32)
    return ((u + (((u >> 16) & 1) + np.uint32(0x7FFF))) & np.uint32(0xFFFF0000)).view(np.float32)


def linear_inputs(seed, n):
    rng = np.random.default_rng(seed * 1_000_003 + n)
    x = rng.uniform(-1, 1, (N, n)).astype(np.float32)
    w = rng.normal(0, 1 / np.sqrt(n), (OUT, n)).astype(np.float32)
    return x, w, x.astype(np.float64) @ w.astype(np.float64).T


def kan_inputs(seed, ni):
    rng = np.random.default_rng(seed * 1_000_003 + 7 * ni + 1)
    x = rng.uniform(-1, 1, (N, ni)).astype(np.float32)
    p = rng.uniform(-np.pi / 2, np.pi / 2, (OUT, ni, G)).astype(np.float32)
    a = rng.normal(0, 1 / np.sqrt(ni * G), (OUT, ni, G)).astype(np.float32)
    k = np.arange(1, G + 1, dtype=np.float32)
    exact = np.zeros((N, OUT))
    for s in range(0, N, KAN_CHUNK):
        z = x[s:s + KAN_CHUNK, None, :, None].astype(np.float64) * k + p[None].astype(np.float64)
        exact[s:s + KAN_CHUNK] = np.einsum("bcik,cik->bc", np.cos(z), a.astype(np.float64))
    return x, p, a, k, exact


def rel_mae(out, exact):
    return float(np.abs(np.asarray(out, np.float64) - exact).mean() / np.abs(exact).mean())


# ------------------------------------------------------------------ simulator (local)

def sim_linear(x, w, mode):
    xb, wb = bf16(x), bf16(w)
    if mode == "f32acc":
        return bf16(xb @ wb.T)
    import experiments.cl68_linear_fprop_accumulation.run as cl68
    return cl68.seq_dot(xb, wb)


def sim_kan(x, p, a, k):
    import experiments.cl67_accumulation_aware_training.run as cl67
    out = np.empty((N, OUT), np.float32)
    for s in range(0, N, KAN_CHUNK):
        z = bf16(x[s:s + KAN_CHUNK])[:, None, :, None] * k + bf16(p)[None]
        t = bf16(bf16(np.cos(z)) * bf16(a)[None]).reshape(len(z), OUT, -1)
        out[s:s + KAN_CHUNK] = cl67.seq_sum(t)
    return out


def run_simulator():
    rows = []
    for seed in SEEDS:
        for n in LINEAR_FANIN:
            x, w, exact = linear_inputs(seed, n)
            rows.append({"primitive": "linear", "seed": seed, "terms": n,
                         "sim_f32acc": rel_mae(sim_linear(x, w, "f32acc"), exact),
                         "sim_seq": rel_mae(sim_linear(x, w, "seq"), exact)})
        for ni in KAN_INPUTS:
            x, p, a, k, exact = kan_inputs(seed, ni)
            rows.append({"primitive": "kan", "seed": seed, "terms": ni * G, "sim_seq": rel_mae(sim_kan(x, p, a, k), exact)})
    return rows


# ------------------------------------------------------------------ Q.ANT (Actions)

def run_qant():
    import qant_native_computing_toolkit.ai as q
    from ml_dtypes import bfloat16 as BF

    def call(fn):
        try:
            return fn(), None
        except Exception as e:  # record and continue: one failing size must not lose the rest
            return None, f"{type(e).__name__}: {e}"[:300]

    rows = []
    for seed in SEEDS:
        for n in LINEAR_FANIN:
            x, w, exact = linear_inputs(seed, n)
            out, err = call(lambda: np.asarray(q.linear_fprop(x.astype(BF), w.astype(BF)), np.float32))
            rows.append({"primitive": "linear", "seed": seed, "terms": n,
                         "qant": None if out is None else rel_mae(out.reshape(N, OUT), exact), "error": err})
        for ni in KAN_INPUTS:
            x, p, a, k, exact = kan_inputs(seed, ni)

            def kan():
                parts = [np.asarray(q.calc_kan_layer_fprop(x[s:s + KAN_CHUNK].astype(BF), p.astype(BF), a.astype(BF),
                                                          k.astype(BF)), np.float32) for s in range(0, N, KAN_CHUNK)]
                return np.concatenate(parts)
            out, err = call(kan)
            rows.append({"primitive": "kan", "seed": seed, "terms": ni * G,
                         "qant": None if out is None else rel_mae(out, exact), "error": err})

    rng = np.random.default_rng(70)
    vals = bf16(np.concatenate([np.linspace(-12, 12, 4000), rng.normal(0, 3, 4000)]).astype(np.float32)).reshape(8, -1)
    acts = {}

    def compare(name, q_out, candidate):
        q_out = np.asarray(q_out, np.float32)
        acts[name] = {"exact_match_fraction": float((q_out == candidate).mean()),
                      "max_abs_diff": float(np.abs(q_out - candidate).max()), "n": int(candidate.size)}

    out, err = call(lambda: q.relu_fprop(vals.astype(BF)))
    compare("relu_fprop vs max(x,0)", out, np.maximum(vals, 0)) if err is None else acts.update({"relu_fprop": err})
    out, err = call(lambda: q.sigmoid_fprop(vals.astype(BF)))
    if err is None:
        compare("sigmoid_fprop vs bf16(sigmoid_f32)", out, bf16(1 / (1 + np.exp(-vals))))
        compare("sigmoid_fprop vs bf16(sigmoid_f64)", out, bf16((1 / (1 + np.exp(-vals.astype(np.float64)))).astype(np.float32)))
    else:
        acts["sigmoid_fprop"] = err
    logits = bf16(rng.normal(0, 3, (500, 10)).astype(np.float32))
    # softmax_fprop rejects batched input ("does not support batched input"), so call it per row, as Exp001/003/009 did.
    out, err = call(lambda: np.stack([np.asarray(q.softmax_fprop(row.astype(BF)), np.float32).reshape(-1) for row in logits]))
    if err is None:
        e = np.exp(logits - logits.max(1, keepdims=True))
        compare("softmax_fprop vs bf16(softmax_f32)", out, bf16(e / e.sum(1, keepdims=True)))
        compare("softmax_fprop argmax vs logits argmax", out.argmax(1), logits.argmax(1))
    else:
        acts["softmax_fprop"] = err
    a_ = bf16(rng.normal(0, 2, (64, 10)).astype(np.float32))
    b_ = bf16(rng.normal(0, 2, 10).astype(np.float32))
    out, err = call(lambda: q.add_bias_fprop(a_.astype(BF), b_.astype(BF)))
    compare("add_bias_fprop vs bf16(a+b)", out, bf16(a_ + b_)) if err is None else acts.update({"add_bias_fprop": err})
    return rows, acts


def summarize(rows, key):
    out = {}
    for prim in ("linear", "kan"):
        for t in sorted({r["terms"] for r in rows if r["primitive"] == prim}):
            vs = [r[key] for r in rows if r["primitive"] == prim and r["terms"] == t and r.get(key) is not None]
            if vs:
                out[f"{prim}_{t}"] = float(np.mean(vs))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=("simulator", "qant"), required=True)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    if args.backend == "simulator":
        rows = run_simulator()
        res = {"experiment": "CL-70", "backend": "simulator", "rows": rows,
               "summary": {k: summarize(rows, k) for k in ("sim_f32acc", "sim_seq")}}
        out = args.out or HERE / "results_simulator.json"
    else:
        rows, acts = run_qant()
        res = {"experiment": "CL-70", "backend": "qant-cpu", "rows": rows, "summary": {"qant": summarize(rows, "qant")},
               "activations": acts}
        out = args.out or HERE / "results_qant.json"
        print("SUMMARY_JSON " + json.dumps(res, sort_keys=True))
    out.write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")
    for k, s in res["summary"].items():
        print(k, {t: f"{v:.2e}" for t, v in s.items()})
    if "activations" in res:
        print(json.dumps(res["activations"], indent=1))


if __name__ == "__main__":
    main()
