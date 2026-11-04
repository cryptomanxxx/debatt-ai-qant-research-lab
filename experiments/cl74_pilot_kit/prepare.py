"""Train the three pilot workloads once and freeze their weights and reference outputs (Tier 0).

Needs the CL-58 Fashion-MNIST download (Tier 1 on a fresh checkout). About 9 minutes.
Run from the repository root: python -m experiments.cl74_pilot_kit.prepare
"""
from __future__ import annotations

import experiments.cl58_qant_quantization_aware_training.run as cl58  # sets single-threaded BLAS first

import multiprocessing as mp
from pathlib import Path

import numpy as np

import experiments.cl69_qant_model_zoo.run as cl69
import experiments.cl72_application_mapping.run as cl72
import experiments.cl73_search_embeddings.run as cl73
from experiments.cl74_pilot_kit import kit

HERE = Path(__file__).resolve().parent
SEED = 11
N_REF_IMAGES, N_REF_QUERIES, N_BENCH_SEQ = 1000, 256, 256


def train_mlp():
    x_tr, y_tr, _, _ = cl58.load_fashion_mnist(cl58.HERE / "data")
    spec, act = cl69.ZOO["mlp_h256x2"]
    m = cl69.Net(SEED, spec, act)
    cl69.train(m, SEED, x_tr, y_tr)
    return {f"mlp.W{i}": L.p["W"].T.copy() for i, L in enumerate(m.layers)} | \
           {f"mlp.b{i}": L.p["b"].copy() for i, L in enumerate(m.layers)}, m


def train_cnn():
    x_tr, y_tr, _, _ = cl58.load_fashion_mnist(cl58.HERE / "data")
    m = cl72.CNN(SEED)
    rng = np.random.default_rng(SEED + 1)
    for _ in range(3):
        order = rng.permutation(len(x_tr))
        for s in range(0, len(order), 128):
            idx = order[s:s + 128]
            m.train_batch(x_tr[idx], y_tr[idx])
    return {f"cnn.{k}": (v.T.copy() if k.endswith("_W") else v.copy()) for k, v in m.p.items()}, m


def train_encoder():
    paras = cl73.load_corpus()
    train_idx, _, _, _ = cl73.build_task(paras)
    vocab = cl73.Vocab(paras, train_idx)
    p = cl73.init_params(SEED)
    cl73.train(p, paras, train_idx, vocab, SEED)
    return {f"enc.{k}": v.copy() for k, v in p.items()}, p


def _job(name):
    return name, {"mlp": train_mlp, "cnn": train_cnn, "encoder": train_encoder}[name]()[0]


def bench_sequences(paras, vocab):
    """Fixed 128-token inputs (CLS + 127 tokens) cut from the concatenated corpus, as at BERT-128 length."""
    stream = [t for w in paras for t in vocab.encode(w)[1:]]
    return np.array([[cl73.CLS] + stream[i * 127:(i + 1) * 127] for i in range(N_BENCH_SEQ)], np.int64)


def main():
    with mp.get_context("fork").Pool(3) as pool:
        W = {k: v for _, d in pool.map(_job, ("mlp", "cnn", "encoder")) for k, v in d.items()}
    W = {k: np.asarray(v, np.float32) for k, v in W.items()}
    np.savez_compressed(HERE / "weights.npz", **W)

    _, _, x_te, y_te = cl58.load_fashion_mnist(cl58.HERE / "data")
    images = np.round(x_te[:N_REF_IMAGES] * 255).astype(np.uint8)
    paras = cl73.load_corpus()
    train_idx, _, queries, index_docs = cl73.build_task(paras)
    vocab = cl73.Vocab(paras, train_idx)
    R = {"images": images, "labels": y_te[:N_REF_IMAGES].astype(np.uint8),
         "enc_query_ids": cl73.pad([vocab.encode(q) for _, q, _ in queries[:N_REF_QUERIES]]),
         "enc_doc_ids": cl73.pad([vocab.encode(d) for d in index_docs]),
         "enc_bench_ids": bench_sequences(paras, vocab)}
    X = images.astype(np.float32) / 255.0
    for be in (kit.Float32(), kit.QantSim()):
        for wl in ("mlp", "cnn"):
            Wp = kit.prepared(be, W, wl)
            R[f"{wl}_{be.name}"] = np.concatenate([kit.run_workload(be, Wp, wl, X[s:s + 250]) for s in range(0, len(X), 250)])
        Wp = kit.prepared(be, W, "encoder")
        eq = np.concatenate([kit.encoder(be, Wp, R["enc_query_ids"][s:s + 128]) for s in range(0, N_REF_QUERIES, 128)])
        ed = np.concatenate([kit.encoder(be, Wp, R["enc_doc_ids"][s:s + 128]) for s in range(0, len(index_docs), 128)])
        R[f"enc_top1_{be.name}"] = (eq @ ed.T).argmax(1)
        R[f"enc_query_emb_{be.name}"] = eq
    np.savez_compressed(HERE / "reference.npz", **R)
    print({k: v.shape for k, v in R.items()})


if __name__ == "__main__":
    main()
