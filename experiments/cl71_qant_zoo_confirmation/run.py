"""CL-71: Q.ANT CPU-backend confirmation of the CL-69 model zoo (phase D, Tier 2 owner-approved).

  --mode qant     (Actions, Q.ANT image) train the CL-69 models, evaluate each on the Q.ANT
                  CPU backend, save per-image reference and Q.ANT labels (also in the log).
  --mode analyze  (local) retrain the same models, check that the reference labels equal the
                  Actions ones, add the simulator labels (CL-69 path) and compute per-image
                  agreement statistics.
See PREDICTIONS.md.
"""
from __future__ import annotations

import experiments.cl69_qant_model_zoo.run as cl69  # sets single-threaded BLAS first

import argparse
import base64
import hashlib
import json
import multiprocessing as mp
import statistics
import zlib
from pathlib import Path

import numpy as np

cl58 = cl69.cl58
HERE = Path(__file__).resolve().parent
SEEDS = (11, 22, 33)
MODELS = ("mlp_h256x3", "mlp_sig_h256", "kan_g8", "kan_g4", "lin128_kanhead", "kan32_linhead")
ARMS = {"mlp_h256x3": [("mlp_h256x3", 1)], "mlp_sig_h256": [("mlp_sig_h256", 1)], "kan_g8": [("kan_g8", 1)],
        "kan_g4": [("kan_g4", 1), ("kan_g4_split16", 16)], "lin128_kanhead": [("lin128_kanhead", 1)],
        "kan32_linhead": [("kan32_linhead", 1)]}


class Qant:
    def __init__(self):
        import qant_native_computing_toolkit.ai as q
        from ml_dtypes import bfloat16
        self.q, self.bf = q, bfloat16

    def f(self, a):
        return np.asarray(a, np.float32)

    def bias(self, o, b):
        return self.f(self.q.add_bias_fprop(self.f(o).astype(self.bf), b.astype(self.bf)))

    def layer(self, L, z, calls):
        bf = self.bf
        if isinstance(L, cl69.Linear):
            return self.bias(self.q.linear_fprop(self.f(z).astype(bf), L.p["W"].astype(bf)), L.p["b"])
        total = np.zeros((len(z), L.p["b"].size), np.float32)
        for c in range(calls):
            lo, hi = c * L.ni // calls, (c + 1) * L.ni // calls
            total = total + self.f(self.q.calc_kan_layer_fprop(self.f(z[:, lo:hi]).astype(bf), L.p["phi"][:, lo:hi].astype(bf),
                                                               L.p["A"][:, lo:hi].astype(bf), L.k.astype(bf)))
        return self.bias(total, L.p["b"])

    def act(self, name, z):
        fn = self.q.relu_fprop if name == "relu" else self.q.sigmoid_fprop
        return self.f(fn(self.f(z).astype(self.bf)))

    def logits(self, model, X, calls=1):
        z = X
        for i, L in enumerate(model.layers):
            z = self.layer(L, z, calls if isinstance(L, cl69.KAN) else 1)
            if i < len(model.layers) - 1 and model.act:
                z = self.act(model.act, z)
        return z


def predict(fn, X, batch=250):
    return np.concatenate([np.asarray(fn(X[s:s + batch])).argmax(1) for s in range(0, len(X), batch)]).astype(np.uint8)


def trained(name, seed, data_dir, n_train=None):
    x_tr, y_tr, x_te, y_te = cl58.load_fashion_mnist(Path(data_dir))
    if n_train:
        x_tr, y_tr = x_tr[:n_train], y_tr[:n_train]
    spec, act = cl69.ZOO[name]
    model = cl69.Net(seed, spec, act)
    cl69.train(model, seed, x_tr, y_tr)
    return model, x_te, y_te


def qant_job(args):
    name, seed, data_dir, n_train, n_test = args
    model, x_te, y_te = trained(name, seed, data_dir, n_train)
    x_te, y_te = x_te[:n_test], y_te[:n_test]
    be = Qant()
    ref = predict(model.logits, x_te)
    out = []
    for arm, calls in ARMS[name]:
        q = predict(lambda X: be.logits(model, X, calls), x_te)
        out.append(({"arm": arm, "seed": seed, "float_accuracy": float((ref == y_te).mean()),
                     "qant_accuracy": float((q == y_te).mean()), "disagreements": int((ref != q).sum())},
                    {f"{arm}_s{seed}_ref": ref, f"{arm}_s{seed}_qant": q}))
    return out


def pack(preds):
    keys = sorted(preds)
    blob = b"".join(preds[k].tobytes() for k in keys)
    return keys, base64.b64encode(zlib.compress(blob, 9)).decode(), hashlib.sha256(blob).hexdigest()


def unpack(p):
    raw = zlib.decompress(base64.b64decode(p["data"]))
    assert hashlib.sha256(raw).hexdigest() == p["sha256_raw"]
    arr = np.frombuffer(raw, np.uint8).reshape(len(p["keys"]), -1)
    return dict(zip(p["keys"], arr))


def run_qant(data_dir, processes, n_train=None, n_test=None):
    cl58.load_fashion_mnist(data_dir)
    jobs = [(m, s, str(data_dir), n_train, n_test or 10000) for m in ("kan32_linhead", "kan_g8", "lin128_kanhead", "kan_g4",
                                                                       "mlp_h256x3", "mlp_sig_h256") for s in SEEDS]
    with mp.get_context("fork").Pool(processes) as pool:
        res = [x for xs in pool.map(qant_job, jobs) for x in xs]
    rows = [r for r, _ in res]
    preds = {k: v for _, p in res for k, v in p.items()}
    keys, data, digest = pack(preds)
    return {"experiment": "CL-71", "backend": "qant-cpu", "rows": rows,
            "predictions": {"keys": keys, "data": data, "sha256_raw": digest, "encoding": "base64(zlib(uint8 concat in key order))"}}


def analyze(qant_file, data_dir, processes):
    qr = json.loads(Path(qant_file).read_text())
    qp = unpack(qr["predictions"])
    cl69.cl67.lib()
    jobs = [(m, s, str(data_dir)) for m in MODELS for s in SEEDS]
    with mp.get_context("fork").Pool(processes) as pool:
        local = [x for xs in pool.map(sim_job, jobs) for x in xs]
    rows = []
    for arm, seed, ref, sim, y in local:
        q, qref = qp[f"{arm}_s{seed}_qant"], qp[f"{arm}_s{seed}_ref"]
        fq, fs = ref != q, ref != sim
        inter = int((fq & fs).sum())
        rows.append({"arm": arm, "seed": seed, "reference_identical_to_actions": bool((ref == qref).all()),
                     "float_accuracy": float((ref == y).mean()), "qant_accuracy": float((q == y).mean()),
                     "sim_accuracy": float((sim == y).mean()), "qant_disagreements": int(fq.sum()),
                     "sim_disagreements": int(fs.sum()), "sim_vs_qant": int((sim != q).sum()),
                     "flipped_both": inter, "recall": inter / max(int(fq.sum()), 1),
                     "precision": inter / max(int(fs.sum()), 1)})
    arms = [a for m in MODELS for a, _ in ARMS[m]]
    summary = {a: {k: statistics.fmean(r[k] for r in rows if r["arm"] == a) for k in rows[0] if k not in ("arm", "seed")}
               for a in arms}
    rho = cl69.spearman([summary[a]["sim_disagreements"] for a in arms], [summary[a]["qant_disagreements"] for a in arms])
    return {"experiment": "CL-71", "rows": rows, "summary": summary, "spearman_sim_vs_qant_disagreements": rho,
            "all_reference_labels_identical": all(r["reference_identical_to_actions"] for r in rows)}


def sim_job(args):
    name, seed, data_dir = args
    model, x_te, y_te = trained(name, seed, data_dir)
    ref = predict(model.logits, x_te)
    return [(arm, seed, ref, predict(lambda X: model.qant_logits(X, calls), x_te), y_te) for arm, calls in ARMS[name]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("qant", "analyze"), required=True)
    ap.add_argument("--data-dir", type=Path, default=cl58.HERE / "data")
    ap.add_argument("--processes", type=int, default=2)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--qant-results", type=Path, default=HERE / "results_qant.json")
    args = ap.parse_args()
    if args.mode == "qant":
        res = run_qant(args.data_dir, args.processes)
        text = json.dumps(res, indent=1, sort_keys=True) + "\n"
        (args.out or HERE / "results_qant.json").write_text(text)
        print("ROWS_JSON " + json.dumps(res["rows"], sort_keys=True))
        print("PREDICTIONS_SHA256 " + res["predictions"]["sha256_raw"])
        print("PREDICTIONS_KEYS " + json.dumps(res["predictions"]["keys"]))
        d = res["predictions"]["data"]
        for i in range(0, len(d), 4000):
            print(f"PREDICTIONS_B64 {i // 4000:04d} {d[i:i + 4000]}")
        for a in sorted({r["arm"] for r in res["rows"]}):
            rs = [r for r in res["rows"] if r["arm"] == a]
            print(f"{a:16s} float {statistics.fmean(r['float_accuracy'] for r in rs):.4f} qant {statistics.fmean(r['qant_accuracy'] for r in rs):.4f}"
                  f" dis {statistics.fmean(r['disagreements'] for r in rs):.1f}")
    else:
        res = analyze(args.qant_results, args.data_dir, args.processes)
        (args.out or HERE / "results.json").write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")
        print("reference labels identical to Actions:", res["all_reference_labels_identical"])
        for a, s in res["summary"].items():
            print(f"{a:16s} qant {s['qant_accuracy']:.4f} dis Q {s['qant_disagreements']:6.1f} S {s['sim_disagreements']:6.1f}"
                  f" S-vs-Q {s['sim_vs_qant']:5.1f} recall {s['recall']:.2f} precision {s['precision']:.2f}")
        print("Spearman:", round(res["spearman_sim_vs_qant_disagreements"], 3))


if __name__ == "__main__":
    main()
