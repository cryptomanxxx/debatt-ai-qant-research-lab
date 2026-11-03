"""CL-66: Q.ANT CPU-backend confirmation of the accumulation simulator and the CL-65 design.

Tier 2, owner-approved: runs once in the Q.ANT research image via the manual workflow
.github/workflows/cl58-qant-qat.yml (experiment: cl66). Trains the CL-58 monolithic
grid-4 model and the CL-64/65 28x8 hierarchical model, then evaluates each on the
Q.ANT CPU backend and with the CL-62 simulator, saving per-image predictions.

Run from the repository root:
    python -m experiments.cl66_qant_confirmation.run --backend qant      # in the Q.ANT image
    python -m experiments.cl66_qant_confirmation.run --backend simulator # local plumbing check
"""
from __future__ import annotations

import experiments.cl65_hierarchical_grouped_head.run as cl65  # sets single-threaded BLAS first

import argparse
import base64
import hashlib
import json
import multiprocessing as mp
import statistics
import zlib
from pathlib import Path

import numpy as np

cl64, cl63, cl58 = cl65.cl64, cl65.cl63, cl65.cl58
cl62 = cl64.cl62
bf16 = cl58.bf16
HERE = Path(__file__).resolve().parent
SEEDS = cl58.SEEDS
HEAD_CALLS = 8
ARMS = ("mono", "h28x8_head1", "h28x8_head8")


# ------------------------------------------------------------------ backends

class SimBackend:
    """CL-62 simulator: one Q.ANT call = bf16 operands, sequential bf16 sum, bf16 output."""

    @staticmethod
    def kan(X, phi, A, k):
        return cl63.kan_sim(X, phi, A, k, "seq")

    @staticmethod
    def bias(o, b):
        return bf16(bf16(o) + bf16(b))


class QantBackend:
    def __init__(self):
        import qant_native_computing_toolkit.ai as q_ai
        from ml_dtypes import bfloat16
        self.q, self.bf = q_ai, bfloat16

    def kan(self, X, phi, A, k):
        bf = self.bf
        out = self.q.calc_kan_layer_fprop(np.asarray(X, np.float32).astype(bf), phi.astype(bf), A.astype(bf), k.astype(bf))
        return np.asarray(out, dtype=np.float32)

    def bias(self, o, b):
        bf = self.bf
        return np.asarray(self.q.add_bias_fprop(np.asarray(o, np.float32).astype(bf), b.astype(bf)), dtype=np.float32)


# ------------------------------------------------------------------ evaluation paths

def mono_logits(be, model, X):
    return be.bias(be.kan(X, model.phi, model.A, model.k), model.b)


def hier_logits(be, model, X, head_calls):
    p = model.p
    hs = [be.bias(be.kan(X[:, i * model.chunk:(i + 1) * model.chunk], p["s1_phi"][i], p["s1_A"][i], cl64.K), p["s1_b"][i])
          for i in range(model.g)]
    H = np.concatenate(hs, axis=1).astype(np.float32)
    if head_calls == 1:
        return be.bias(be.kan(H, p["s2_phi"], p["s2_A"], cl64.K), p["s2_b"])
    n = H.shape[1]
    total = np.zeros((len(H), cl58.C), np.float32)
    for j in range(head_calls):
        lo, hi = j * n // head_calls, (j + 1) * n // head_calls
        total = total + be.kan(H[:, lo:hi], p["s2_phi"][:, lo:hi], p["s2_A"][:, lo:hi], cl64.K)
    return be.bias(total, p["s2_b"])


def predict(fn, X, batch=250):
    return np.concatenate([fn(X[s:s + batch]).argmax(1) for s in range(0, len(X), batch)]).astype(np.uint8)


def flip_stats(ref, sim, q):
    fq, fs = ref != q, ref != sim
    inter, union = int((fq & fs).sum()), int((fq | fs).sum())
    return {"dis_ref_qant": int(fq.sum()), "dis_ref_sim": int(fs.sum()), "dis_sim_qant": int((sim != q).sum()),
            "flipped_both": inter, "recall": inter / max(int(fq.sum()), 1), "precision": inter / max(int(fs.sum()), 1),
            "jaccard": inter / max(union, 1)}


# ------------------------------------------------------------------ jobs

def job(args):
    kind, seed, data_dir, backend, n_train, n_test = args
    x_tr, y_tr, x_te, y_te = cl58.load_fashion_mnist(Path(data_dir))
    if n_train:
        x_tr, y_tr = x_tr[:n_train], y_tr[:n_train]
    if n_test:
        x_te, y_te = x_te[:n_test], y_te[:n_test]
    be_q = QantBackend() if backend == "qant" else SimBackend()
    be_s = SimBackend()
    out = []
    if kind == "mono":
        model = cl62.train(seed, 4, "exact_cosine", x_tr, y_tr)
        ref = model.predict(x_te).astype(np.uint8)
        paths = {"mono": lambda be: (lambda X: mono_logits(be, model, X))}
    else:
        model = cl64.Hier(seed, 28, 8)
        rng = np.random.default_rng(seed + 1)
        for _ in range(cl64.EPOCHS):
            order = rng.permutation(len(x_tr))
            for s in range(0, len(order), cl64.BATCH):
                idx = order[s:s + cl64.BATCH]
                model.train_batch(x_tr[idx], y_tr[idx])
        ref = cl64.predict(model.logits, x_te).astype(np.uint8)
        paths = {f"h28x8_head{h}": (lambda be, h=h: (lambda X: hier_logits(be, model, X, h))) for h in (1, HEAD_CALLS)}
    for arm, make in paths.items():
        sim = predict(make(be_s), x_te)
        q = predict(make(be_q), x_te)
        row = {"arm": arm, "seed": seed, "reference_accuracy": float((ref == y_te).mean()),
               "sim_accuracy": float((sim == y_te).mean()), "qant_accuracy": float((q == y_te).mean())}
        row.update(flip_stats(ref, sim, q))
        out.append((row, {"ref": ref, "sim": sim, "qant": q}))
    return out


def pack(preds):
    """Concatenate uint8 label arrays in a fixed order, zlib-compress, base64-encode."""
    order = sorted(preds)
    blob = b"".join(preds[k].tobytes() for k in order)
    return order, base64.b64encode(zlib.compress(blob, 9)).decode(), hashlib.sha256(blob).hexdigest()


def run(data_dir, backend, processes, n_train=None, n_test=None):
    cl58.load_fashion_mnist(data_dir)
    jobs = [(k, s, str(data_dir), backend, n_train, n_test) for k in ("mono", "hier") for s in SEEDS]
    with mp.get_context("fork").Pool(processes) as pool:
        results = [x for xs in pool.map(job, jobs) for x in xs]
    rows = [r for r, _ in results]
    preds = {f"{r['arm']}_s{r['seed']}_{name}": arr for r, p in results for name, arr in p.items()}
    keys, packed, digest = pack(preds)
    summary = {}
    for arm in ARMS:
        rs = [r for r in rows if r["arm"] == arm]
        summary[arm] = {k: statistics.fmean(r[k] for r in rs) for k in rs[0] if k not in ("arm", "seed")}
    return {"experiment": "CL-66", "backend": backend, "dataset": "FashionMNIST", "seeds": list(SEEDS),
            "head_calls": HEAD_CALLS, "n_test": n_test or 10000, "numpy": np.__version__,
            "rows": rows, "summary": summary,
            "predictions": {"keys": keys, "dtype": "uint8", "length_each": int(len(next(iter(preds.values())))),
                            "encoding": "base64(zlib(concat in key order))", "sha256_raw": digest, "data": packed}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=("qant", "simulator"), required=True)
    ap.add_argument("--data-dir", type=Path, default=cl58.HERE / "data")
    ap.add_argument("--processes", type=int, default=2)
    ap.add_argument("--n-train", type=int, default=None)
    ap.add_argument("--n-test", type=int, default=None)
    ap.add_argument("--out", type=Path, default=HERE / "results_qant.json")
    args = ap.parse_args()
    res = run(args.data_dir, args.backend, args.processes, args.n_train, args.n_test)
    text = json.dumps(res, indent=1, sort_keys=True) + "\n"
    args.out.write_text(text)
    print("results sha256:", hashlib.sha256(text.encode()).hexdigest())
    print("SUMMARY_JSON " + json.dumps({"rows": res["rows"], "summary": res["summary"]}, sort_keys=True))
    print("PREDICTIONS_SHA256 " + res["predictions"]["sha256_raw"])
    print("PREDICTIONS_KEYS " + json.dumps(res["predictions"]["keys"]))
    data = res["predictions"]["data"]
    for i in range(0, len(data), 4000):  # chunked so the job log keeps every line intact
        print(f"PREDICTIONS_B64 {i // 4000:04d} {data[i:i + 4000]}")
    for arm, s in res["summary"].items():
        print(f"{arm:12s} ref {s['reference_accuracy']:.4f} qant {s['qant_accuracy']:.4f} sim {s['sim_accuracy']:.4f}"
              f" dQ {s['dis_ref_qant']:.1f} dS {s['dis_ref_sim']:.1f} dSQ {s['dis_sim_qant']:.1f}"
              f" recall {s['recall']:.2f} precision {s['precision']:.2f} jaccard {s['jaccard']:.2f}")


if __name__ == "__main__":
    main()
