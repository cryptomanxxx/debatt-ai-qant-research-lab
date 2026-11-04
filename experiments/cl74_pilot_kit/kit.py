"""Phase E pilot kit: one benchmark for any backend that provides a matrix product.

Standalone: needs only NumPy (and, for the Q.ANT backend, Q.ANT's toolkit and ml_dtypes). It reads the fixed
weights and reference outputs written by prepare.py, so the pilot needs no training and no dataset download.

    python kit.py check --backend float32|qant_sim|qant_toolkit
    python kit.py bench --backend float32 --threads 1 [--watts W] [--out file.json]

`--watts` is a measured average power during the benchmark, from an instrument on the pilot. Without it,
energy is reported as NOT_MEASURED. Never pass an estimated or nameplate value.
"""
from __future__ import annotations

import os
import sys


def _threads_from_argv(argv):
    """Read --threads N or --threads=N before NumPy is imported (BLAS reads its thread count at import)."""
    for i, a in enumerate(argv):
        if a == "--threads" and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith("--threads="):
            return a.split("=", 1)[1]
    return "1"


THREADS = _threads_from_argv(sys.argv)


for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = THREADS  # must precede the NumPy import

import argparse
import json
import math
import platform
import statistics
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BATCHES = (1, 8, 64, 256)


def bf16(a):
    """Round float32 to the nearest bfloat16 (ties to even), returned as float32."""
    a = np.ascontiguousarray(a, dtype=np.float32)
    u = a.view(np.uint32)
    return ((u + (((u >> 16) & 1) + np.uint32(0x7FFF))) & np.uint32(0xFFFF0000)).view(np.float32)


# ------------------------------------------------------------------ backends
# Weights are stored in "x @ W" layout (fan-in first). A backend prepares each weight once on the host,
# then multiplies activations by it. mm_dyn multiplies two activation tensors (attention QK^T and AV).

class Float32:
    name = "float32"

    def prepare(self, W):
        return np.ascontiguousarray(W, np.float32)

    def mm(self, x, Wp):
        return x @ Wp

    def mm_dyn(self, a, b):
        return a @ b

    def bias(self, o, b):
        return o + b


class QantSim(Float32):
    """CPU-backend model of linear_fprop (CL-70): bf16 operands and output, float32 accumulation."""
    name = "qant_sim"

    def prepare(self, W):
        return bf16(W)

    def mm(self, x, Wp):
        return bf16(bf16(x) @ Wp)

    def mm_dyn(self, a, b):
        return bf16(bf16(a) @ bf16(b))

    def bias(self, o, b):
        return bf16(o + bf16(b))


class QantToolkit(Float32):
    """Q.ANT Native Computing Toolkit. Not executed in CL-74: running it is Tier 2 (external system)."""
    name = "qant_toolkit"

    def __init__(self):
        import qant_native_computing_toolkit.ai as q
        from ml_dtypes import bfloat16
        self.q, self.bf = q, bfloat16

    def prepare(self, W):
        return np.ascontiguousarray(np.asarray(W, np.float32).T).astype(self.bf)  # linear_fprop takes (out, in)

    def mm(self, x, Wp):
        shape = x.shape
        o = self.q.linear_fprop(np.asarray(x, np.float32).reshape(-1, shape[-1]).astype(self.bf), Wp)
        return np.asarray(o, np.float32).reshape(*shape[:-1], -1)

    def mm_dyn(self, a, b):
        lead = a.shape[:-2]
        a2 = a.reshape(-1, *a.shape[-2:])
        b2 = b.reshape(-1, *b.shape[-2:])
        outs = [np.asarray(self.q.linear_fprop(x.astype(self.bf), np.ascontiguousarray(y.T).astype(self.bf)), np.float32)
                for x, y in zip(a2, b2)]
        return np.stack(outs).reshape(*lead, a.shape[-2], b.shape[-1])

    def bias(self, o, b):
        shape = o.shape
        r = self.q.add_bias_fprop(np.asarray(o, np.float32).reshape(-1, shape[-1]).astype(self.bf),
                                  np.asarray(b, np.float32).astype(self.bf))
        return np.asarray(r, np.float32).reshape(shape)


BACKENDS = {"float32": Float32, "qant_sim": QantSim, "qant_toolkit": QantToolkit}


class Timed:
    """Wraps a backend and accumulates the wall time spent inside matrix products and bias adds."""

    def __init__(self, be):
        self.be, self.t = be, 0.0

    def _timed(self, fn, *a):
        t0 = time.perf_counter()
        r = fn(*a)
        self.t += time.perf_counter() - t0
        return r

    def mm(self, x, Wp):
        return self._timed(self.be.mm, x, Wp)

    def mm_dyn(self, a, b):
        return self._timed(self.be.mm_dyn, a, b)

    def bias(self, o, b):
        return self._timed(self.be.bias, o, b)


# ------------------------------------------------------------------ workloads

def relu(x):
    return np.maximum(x, 0)


def mlp(be, W, X):
    z = X
    for i in range(3):
        z = be.bias(be.mm(z, W[f"mlp.W{i}"]), W[f"mlp.b{i}"])
        if i < 2:
            z = relu(z)
    return z


def im2col(x, k):
    B, H, Wd, C = x.shape
    ho, wo = H - k + 1, Wd - k + 1
    return np.concatenate([x[:, i:i + ho, j:j + wo, :] for i in range(k) for j in range(k)], axis=3)


def pool(x):
    B, H, Wd, C = x.shape
    return x.reshape(B, H // 2, 2, Wd // 2, 2, C).mean((2, 4))


def cnn(be, W, X):
    x0 = X.reshape(-1, 28, 28, 1)
    a1 = relu(be.bias(be.mm(im2col(x0, 5), W["cnn.c1_W"]), W["cnn.c1_b"]))
    a2 = relu(be.bias(be.mm(im2col(pool(a1), 5), W["cnn.c2_W"]), W["cnn.c2_b"]))
    return be.bias(be.mm(pool(a2).reshape(len(X), -1), W["cnn.fc_W"]), W["cnn.fc_b"])


ENC_D, ENC_H, ENC_LAYERS = 64, 4, 2


def layernorm(x, g, b):
    mu = x.mean(-1, keepdims=True)
    xc = x - mu
    return xc / np.sqrt((xc * xc).mean(-1, keepdims=True) + 1e-5) * g + b


def gelu(x):
    return 0.5 * x * (1 + np.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * (x * x * x))))  # x**3 is ~70x slower in NumPy


def softmax(s):
    s = s - s.max(-1, keepdims=True)
    e = np.exp(s)
    return e / e.sum(-1, keepdims=True)


def encoder(be, W, ids):
    """CL-73 encoder. Token id 0 is padding. Returns L2-normalised embeddings."""
    B, L = ids.shape
    mask = (ids != 0).astype(np.float32)
    neg = (1.0 - mask)[:, None, None, :] * -1e9
    x = W["enc.E"][ids] + W["enc.P"][:L]
    dh = ENC_D // ENC_H
    split = lambda t: t.reshape(B, L, ENC_H, dh).transpose(0, 2, 1, 3)
    for l in range(ENC_LAYERS):
        h = layernorm(x, W[f"enc.g1{l}"], W[f"enc.be1{l}"])
        q, k, v = (split(be.bias(be.mm(h, W[f"enc.W{n}{l}"]), W[f"enc.b{n}{l}"])) for n in "qkv")
        a = softmax(be.mm_dyn(q, k.transpose(0, 1, 3, 2)) * (1.0 / math.sqrt(dh)) + neg)
        o = be.mm_dyn(a, v).transpose(0, 2, 1, 3).reshape(B, L, ENC_D)
        x = x + be.bias(be.mm(o, W[f"enc.Wo{l}"]), W[f"enc.bo{l}"])
        h2 = layernorm(x, W[f"enc.g2{l}"], W[f"enc.be2{l}"])
        x = x + be.bias(be.mm(gelu(be.bias(be.mm(h2, W[f"enc.W1{l}"]), W[f"enc.b1{l}"])), W[f"enc.W2{l}"]), W[f"enc.b2{l}"])
    xf = layernorm(x, W["enc.gf"], W["enc.bf"])
    pooled = (xf * mask[:, :, None]).sum(1) / mask.sum(1, keepdims=True)
    z = be.bias(be.mm(pooled, W["enc.Wp"]), W["enc.bp"])
    return z / np.sqrt((z * z).sum(1, keepdims=True))


WORKLOADS = {"mlp": mlp, "cnn": cnn, "encoder": encoder}
MATRIX_WEIGHTS = {  # weights that go through backend.prepare (everything else stays on the host)
    "mlp": ["mlp.W0", "mlp.W1", "mlp.W2"],
    "cnn": ["cnn.c1_W", "cnn.c2_W", "cnn.fc_W"],
    "encoder": [f"enc.W{n}{l}" for l in range(ENC_LAYERS) for n in ("q", "k", "v", "o", "1", "2")] + ["enc.Wp"],
}


def macs(workload, seq=128):
    if workload == "mlp":
        return 784 * 256 + 256 * 256 + 256 * 10
    if workload == "cnn":
        return 24 * 24 * 16 * 25 + 8 * 8 * 32 * 400 + 512 * 10
    static = seq * (4 * ENC_D * ENC_D + 2 * ENC_D * 256)
    return ENC_LAYERS * (static + 2 * seq * seq * ENC_D) + ENC_D * ENC_D


def load():
    W = dict(np.load(HERE / "weights.npz"))
    R = dict(np.load(HERE / "reference.npz"))
    return W, R


def prepared(be, W, workload):
    out = dict(W)
    for k in MATRIX_WEIGHTS[workload]:
        out[k] = be.prepare(W[k])
    return out


def run_workload(be, W, workload, X):
    return WORKLOADS[workload](be, W, X)


def inputs(R, workload, n):
    if workload == "encoder":
        return R["enc_bench_ids"][:n]
    return R["images"][:n].astype(np.float32) / 255.0


# ------------------------------------------------------------------ check (fidelity)

def check(be):
    W, R = load()
    out = {}
    for wl in ("mlp", "cnn"):
        Wp = prepared(be, W, wl)
        X = R["images"].astype(np.float32) / 255.0
        y = np.concatenate([run_workload(be, Wp, wl, X[s:s + 250]) for s in range(0, len(X), 250)])
        ref = R[f"{wl}_float32"]
        out[wl] = {"n": len(X), "top1_agreement_float32": float((y.argmax(1) == ref.argmax(1)).mean()),
                   "top1_agreement_qant_sim": float((y.argmax(1) == R[f"{wl}_qant_sim"].argmax(1)).mean()),
                   "max_rel_error_vs_float32": float(np.abs(y - ref).max() / np.abs(ref).max())}
    Wp = prepared(be, W, "encoder")
    eq = np.concatenate([run_workload(be, Wp, "encoder", R["enc_query_ids"][s:s + 128]) for s in range(0, len(R["enc_query_ids"]), 128)])
    ed = np.concatenate([run_workload(be, Wp, "encoder", R["enc_doc_ids"][s:s + 128]) for s in range(0, len(R["enc_doc_ids"]), 128)])
    top1 = (eq @ ed.T).argmax(1)
    out["encoder"] = {"n": len(eq), "top1_agreement_float32": float((top1 == R["enc_top1_float32"]).mean()),
                      "top1_agreement_qant_sim": float((top1 == R["enc_top1_qant_sim"]).mean()),
                      "min_cosine_vs_float32": float((eq * R["enc_query_emb_float32"]).sum(1).min())}
    return out


# ------------------------------------------------------------------ bench (latency, throughput)

def bench_one(be, W, workload, batch, R, min_time=1.0, min_reps=5, max_reps=500):
    X = inputs(R, workload, batch)
    tb = Timed(be)
    t0 = time.perf_counter()
    run_workload(tb, W, workload, X)  # first call after the weights were prepared: includes any lazy device load
    first = time.perf_counter() - t0
    run_workload(tb, W, workload, X)  # warm-up
    times, mtimes, total = [], [], 0.0
    while (total < min_time or len(times) < min_reps) and len(times) < max_reps:
        tb.t = 0.0
        t0 = time.perf_counter()
        run_workload(tb, W, workload, X)
        dt = time.perf_counter() - t0
        times.append(dt)
        mtimes.append(tb.t)
        total += dt
    med = statistics.median(times)
    return {"workload": workload, "batch": batch, "reps": len(times), "first_call_ms": 1e3 * first,
            "latency_ms_median": 1e3 * med,
            "latency_ms_p95": 1e3 * float(np.percentile(times, 95)), "inferences_per_s": batch / med,
            "gmac_per_s": batch * macs(workload) / med / 1e9,
            "matrix_time_share": float(np.sum(mtimes) / np.sum(times))}


def bench(be, threads, watts=None):
    W, R = load()
    rows = []
    for wl in WORKLOADS:
        loads = []
        for _ in range(5):
            t0 = time.perf_counter()
            Wp = prepared(be, W, wl)
            loads.append(time.perf_counter() - t0)
        for b in BATCHES:
            r = bench_one(be, Wp, wl, b, R)
            # host-side conversion only; the kit cannot see when the device loads weights, which happens inside
            # the timed toolkit calls (compare first_call_ms with latency_ms_median for a hint)
            r["host_weight_prep_ms"] = 1e3 * statistics.median(loads)
            r["device_weight_load_ms"] = "NOT_MEASURED"
            r["energy_j_per_inference"] = (watts * r["latency_ms_median"] / 1e3 / b) if watts else "NOT_MEASURED"
            rows.append(r)
    return {"backend": be.name, "threads": int(THREADS), "watts_measured": watts if watts else "NOT_MEASURED",
            "machine": platform.processor() or platform.machine(), "cpu_count": os.cpu_count(),
            "numpy": np.__version__, "rows": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("check", "bench"))
    ap.add_argument("--backend", default="float32", choices=sorted(BACKENDS))
    ap.add_argument("--threads", default="1")
    ap.add_argument("--watts", type=float, default=None, help="measured average power (instrument), never an estimate")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    be = BACKENDS[args.backend]()
    res = check(be) if args.mode == "check" else bench(be, args.threads, args.watts)
    txt = json.dumps(res, indent=2)
    if args.out:
        args.out.write_text(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
