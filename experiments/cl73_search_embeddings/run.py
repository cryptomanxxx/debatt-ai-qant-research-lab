"""CL-73: search embeddings from a small BERT-style encoder on the simulated Q.ANT path (Tier 0).

Run from the repository root: python -m experiments.cl73_search_embeddings.run
See PREDICTIONS.md (pushed before any training) and README.md.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import gzip
import json
import math
import multiprocessing as mp
import statistics
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
D, H, FF, LAYERS, MAXLEN, VOCAB = 64, 4, 256, 2, 128, 4000
PAD, UNK, CLS = 0, 1, 2
TAU = 0.05


def bf16(a):
    """Round float32 to the nearest bfloat16 (ties to even), returned as float32."""
    a = np.ascontiguousarray(a, dtype=np.float32)
    u = a.view(np.uint32)
    return ((u + (((u >> 16) & 1) + np.uint32(0x7FFF))) & np.uint32(0xFFFF0000)).view(np.float32)


# ------------------------------------------------------------------ data

def load_corpus():
    with gzip.open(HERE / "corpus.json.gz", "rt", encoding="utf-8") as fh:
        return [p["words"] for p in json.load(fh)["paragraphs"]]


def random_span(rng, n):
    length = int(rng.integers(12, 21))
    start = int(rng.integers(0, n - length + 1))
    return start, start + length


def build_task(paras, n_queries=4):
    """Fixed split (seed 0). Each held-out paragraph gets n_queries spans; its document drops that span."""
    rng = np.random.default_rng(0)
    order = rng.permutation(len(paras))
    test = sorted(order[: len(paras) // 5].tolist())
    train = sorted(order[len(paras) // 5:].tolist())
    queries = []  # (paragraph index, query words, document words)
    for i in test:
        w = paras[i]
        for _ in range(n_queries):
            s, e = random_span(rng, len(w))
            queries.append((i, w[s:e], w[:s] + w[e:]))
    index_docs = [list(w) for w in paras]
    for i in test:  # the stored document of a held-out paragraph is its first query's document
        index_docs[i] = next(d for j, _, d in queries if j == i)
    return train, test, queries, index_docs


class Vocab:
    def __init__(self, paras, train):
        counts = Counter(w for i in train for w in paras[i])
        self.ids = {w: k + 3 for k, (w, _) in enumerate(counts.most_common(VOCAB))}

    def encode(self, words):
        return [CLS] + [self.ids.get(w, UNK) for w in words[: MAXLEN - 1]]


def pad(seqs):
    L = max(len(s) for s in seqs)
    out = np.zeros((len(seqs), L), dtype=np.int64)
    for k, s in enumerate(seqs):
        out[k, : len(s)] = s
    return out


# ------------------------------------------------------------------ model

def init_params(seed):
    rng = np.random.default_rng(seed)
    n = lambda *s: (rng.standard_normal(s) * 0.02).astype(np.float32)
    p = {"E": n(VOCAB + 3, D), "P": n(MAXLEN, D), "Wp": n(D, D), "bp": np.zeros(D, np.float32),
         "gf": np.ones(D, np.float32), "bf": np.zeros(D, np.float32)}
    for l in range(LAYERS):
        for nm in ("q", "k", "v", "o"):
            p[f"W{nm}{l}"] = n(D, D)
            p[f"b{nm}{l}"] = np.zeros(D, np.float32)
        p[f"W1{l}"], p[f"b1{l}"] = n(D, FF), np.zeros(FF, np.float32)
        p[f"W2{l}"], p[f"b2{l}"] = n(FF, D), np.zeros(D, np.float32)
        for nm in ("1", "2"):
            p[f"g{nm}{l}"], p[f"be{nm}{l}"] = np.ones(D, np.float32), np.zeros(D, np.float32)
    return p


def n_params(p):
    return int(sum(v.size for v in p.values()))


def ln_fwd(x, g, b):
    mu = x.mean(-1, keepdims=True)
    xc = x - mu
    inv = 1.0 / np.sqrt((xc * xc).mean(-1, keepdims=True) + 1e-5)
    xh = xc * inv
    return xh * g + b, (xh, inv, g)


def ln_bwd(dy, c):
    xh, inv, g = c
    dxh = dy * g
    dx = inv * (dxh - dxh.mean(-1, keepdims=True) - xh * (dxh * xh).mean(-1, keepdims=True))
    return dx, (dy * xh).reshape(-1, xh.shape[-1]).sum(0), dy.reshape(-1, dy.shape[-1]).sum(0)


C_GELU = math.sqrt(2 / math.pi)


def gelu(x):
    t = np.tanh(C_GELU * (x + 0.044715 * x ** 3))
    return 0.5 * x * (1 + t), t


def gelu_bwd(dy, x, t):
    return dy * (0.5 * (1 + t) + 0.5 * x * (1 - t * t) * C_GELU * (1 + 3 * 0.044715 * x * x))


def softmax(s):
    s = s - s.max(-1, keepdims=True)
    e = np.exp(s)
    return e / e.sum(-1, keepdims=True)


class Path_:
    """Arithmetic of one execution path. 'f32' is the reference; 'qant' models linear_fprop (CL-70)."""

    def __init__(self, mode):
        self.q = mode != "f32"
        self.bf16_softmax = mode == "qant_bf16softmax"

    def mm(self, a, b):
        if not self.q:
            return a @ b
        return bf16(bf16(a) @ bf16(b))  # bf16 operands, float32 accumulation, bf16 output

    def bias(self, o, b):
        return o + b if not self.q else bf16(o + bf16(b))  # add_bias_fprop is exact on bf16 values

    def softmax(self, s):
        a = softmax(s)
        return bf16(a) if self.bf16_softmax else a


def forward(p, ids, path=Path_("f32"), cache=False):
    B, L = ids.shape
    mask = (ids != PAD).astype(np.float32)
    neg = (1.0 - mask)[:, None, None, :] * -1e9
    x = p["E"][ids] + p["P"][:L]
    dh = D // H
    caches = []
    split = lambda t: t.reshape(B, L, H, dh).transpose(0, 2, 1, 3)
    for l in range(LAYERS):
        h, c1 = ln_fwd(x, p[f"g1{l}"], p[f"be1{l}"])
        q = split(path.bias(path.mm(h, p[f"Wq{l}"]), p[f"bq{l}"]))
        k = split(path.bias(path.mm(h, p[f"Wk{l}"]), p[f"bk{l}"]))
        v = split(path.bias(path.mm(h, p[f"Wv{l}"]), p[f"bv{l}"]))
        s = path.mm(q, k.transpose(0, 1, 3, 2)) * (1.0 / math.sqrt(dh)) + neg  # dynamic product QK^T
        a = path.softmax(s)
        o = path.mm(a, v).transpose(0, 2, 1, 3).reshape(B, L, D)  # dynamic product AV
        x = x + path.bias(path.mm(o, p[f"Wo{l}"]), p[f"bo{l}"])
        h2, c2 = ln_fwd(x, p[f"g2{l}"], p[f"be2{l}"])
        u = path.bias(path.mm(h2, p[f"W1{l}"]), p[f"b1{l}"])
        f, t = gelu(u)
        x = x + path.bias(path.mm(f, p[f"W2{l}"]), p[f"b2{l}"])
        if cache:
            caches.append((h, c1, q, k, v, a, o, h2, c2, u, t, f))
    xf, cf = ln_fwd(x, p["gf"], p["bf"])
    cnt = mask.sum(1, keepdims=True)
    pooled = (xf * mask[:, :, None]).sum(1) / cnt
    z = path.bias(path.mm(pooled, p["Wp"]), p["bp"])
    nrm = np.sqrt((z * z).sum(1, keepdims=True))
    e = z / nrm
    if cache:
        return e, (ids, mask, caches, cf, cnt, pooled, nrm, e)
    return e


def backward(p, de, c):
    ids, mask, caches, cf, cnt, pooled, nrm, e = c
    B, L = ids.shape
    dh = D // H
    g = {k: np.zeros_like(v) for k, v in p.items()}
    dz = (de - e * (e * de).sum(1, keepdims=True)) / nrm
    g["Wp"] += pooled.T @ dz
    g["bp"] += dz.sum(0)
    dpooled = dz @ p["Wp"].T
    dxf = (dpooled / cnt)[:, None, :] * mask[:, :, None]
    dx, g_g, g_b = ln_bwd(dxf, cf)
    g["gf"] += g_g
    g["bf"] += g_b
    merge = lambda t: t.transpose(0, 2, 1, 3).reshape(B, L, D)
    for l in reversed(range(LAYERS)):
        h, c1, q, k, v, a, o, h2, c2, u, t, f = caches[l]
        # feed-forward
        g[f"W2{l}"] += f.reshape(-1, FF).T @ dx.reshape(-1, D)
        g[f"b2{l}"] += dx.reshape(-1, D).sum(0)
        du = gelu_bwd(dx @ p[f"W2{l}"].T, u, t)
        g[f"W1{l}"] += h2.reshape(-1, D).T @ du.reshape(-1, FF)
        g[f"b1{l}"] += du.reshape(-1, FF).sum(0)
        dh2 = du @ p[f"W1{l}"].T
        dxx, gg, gb = ln_bwd(dh2, c2)
        g[f"g2{l}"] += gg
        g[f"be2{l}"] += gb
        dx = dx + dxx
        # attention
        g[f"Wo{l}"] += o.reshape(-1, D).T @ dx.reshape(-1, D)
        g[f"bo{l}"] += dx.reshape(-1, D).sum(0)
        do = (dx @ p[f"Wo{l}"].T).reshape(B, L, H, dh).transpose(0, 2, 1, 3)
        da = do @ v.transpose(0, 1, 3, 2)
        dv = a.transpose(0, 1, 3, 2) @ do
        ds = a * (da - (da * a).sum(-1, keepdims=True)) / math.sqrt(dh)
        dq = ds @ k
        dk = ds.transpose(0, 1, 3, 2) @ q
        dhh = np.zeros_like(h)
        for nm, dt in (("q", dq), ("k", dk), ("v", dv)):
            dt = merge(dt)
            g[f"W{nm}{l}"] += h.reshape(-1, D).T @ dt.reshape(-1, D)
            g[f"b{nm}{l}"] += dt.reshape(-1, D).sum(0)
            dhh += dt @ p[f"W{nm}{l}"].T
        dxx, gg, gb = ln_bwd(dhh, c1)
        g[f"g1{l}"] += gg
        g[f"be1{l}"] += gb
        dx = dx + dxx
    g["P"][:L] += dx.sum(0)
    np.add.at(g["E"], ids, dx)
    return g


def contrastive(eq, ed):
    """Symmetric in-batch softmax loss; returns loss and gradients w.r.t. both embedding sets."""
    S = eq @ ed.T / TAU
    n = len(S)
    loss, dS = 0.0, np.zeros_like(S)
    for M, sgn in ((S, 0), (S.T, 1)):
        P = softmax(M)
        loss += -np.log(P[np.arange(n), np.arange(n)] + 1e-12).mean() / 2
        G = P.copy()
        G[np.arange(n), np.arange(n)] -= 1
        dS += (G if sgn == 0 else G.T) / (2 * n)
    dS /= TAU
    return float(loss), dS @ ed, dS.T @ eq


def train_step(p, opt, q_ids, d_ids, lr=1e-3):
    n = len(q_ids)
    ids = pad(q_ids + d_ids)
    e, c = forward(p, ids, cache=True)
    loss, deq, ded = contrastive(e[:n], e[n:])
    g = backward(p, np.concatenate([deq, ded]), c)
    opt["t"] += 1
    t = opt["t"]
    for k in p:
        opt["m"][k] = 0.9 * opt["m"][k] + 0.1 * g[k]
        opt["v"][k] = 0.999 * opt["v"][k] + 0.001 * g[k] * g[k]
        mh = opt["m"][k] / (1 - 0.9 ** t)
        vh = opt["v"][k] / (1 - 0.999 ** t)
        p[k] -= (lr * mh / (np.sqrt(vh) + 1e-8)).astype(np.float32)
    return loss


def train(p, paras, train_idx, vocab, seed, epochs=40, batch=64):
    rng = np.random.default_rng(seed + 1)
    opt = {"t": 0, "m": {k: np.zeros_like(v) for k, v in p.items()}, "v": {k: np.zeros_like(v) for k, v in p.items()}}
    losses = []
    for _ in range(epochs):
        order = rng.permutation(train_idx)
        ep = []
        for s in range(0, len(order) - batch + 1, batch):
            qs, ds = [], []
            for i in order[s:s + batch]:
                w = paras[i]
                a, b = random_span(rng, len(w))
                qs.append(vocab.encode(w[a:b]))
                ds.append(vocab.encode(w[:a] + w[b:]))
            ep.append(train_step(p, opt, qs, ds))
        losses.append(float(np.mean(ep)))
    return losses


# ------------------------------------------------------------------ evaluation

def embed(p, seqs, path, chunk=128):
    out = []
    for s in range(0, len(seqs), chunk):
        out.append(forward(p, pad(seqs[s:s + chunk]), path))
    return np.concatenate(out)


def score_matrix(eq, edoc, eown, owner):
    """Query × index cosine scores, with each query's own document replaced by its query-specific version."""
    S = eq @ edoc.T
    S[np.arange(len(eq)), owner] = (eq * eown).sum(1)
    return S


def rank_metrics(S, owner):
    own = S[np.arange(len(S)), owner]
    rank = (S > own[:, None]).sum(1)  # 0-based; ties count in the query's favour
    return {"recall_at_1": float((rank == 0).mean()), "recall_at_10": float((rank < 10).mean()),
            "mrr": float((1.0 / (rank + 1)).mean())}


def top10(S):
    return np.argsort(-S, axis=1, kind="stable")[:, :10]


def bm25_scores(index_docs, queries, k1=1.2, b=0.75):
    N = len(index_docs)
    df = Counter(w for d in index_docs for w in set(d))
    idf = {w: math.log(1 + (N - n + 0.5) / (n + 0.5)) for w, n in df.items()}
    avg = statistics.fmean(len(d) for d in index_docs)

    def doc_score(qw, d):
        tf = Counter(d)
        norm = k1 * (1 - b + b * len(d) / avg)
        return sum(idf.get(w, 0.0) * tf[w] * (k1 + 1) / (tf[w] + norm) for w in qw if w in tf)

    tfs = [Counter(d) for d in index_docs]
    S = np.zeros((len(queries), N))
    owner = np.array([i for i, _, _ in queries])
    for r, (i, qw, dw) in enumerate(queries):
        for j, (d, tf) in enumerate(zip(index_docs, tfs)):
            if not set(qw).isdisjoint(tf):
                norm = k1 * (1 - b + b * len(d) / avg)
                S[r, j] = sum(idf.get(w, 0.0) * tf[w] * (k1 + 1) / (tf[w] + norm) for w in qw if w in tf)
        S[r, i] = doc_score(qw, dw)
    return S, owner


def evaluate(p, vocab, queries, index_docs):
    owner = np.array([i for i, _, _ in queries])
    qseq = [vocab.encode(q) for _, q, _ in queries]
    oseq = [vocab.encode(d) for _, _, d in queries]
    dseq = [vocab.encode(d) for d in index_docs]
    emb = {}
    for mode in ("f32", "qant", "qant_bf16softmax"):
        path = Path_(mode)
        emb[mode] = (embed(p, qseq, path), embed(p, dseq, path), embed(p, oseq, path))
    variants = {
        "float32": (emb["f32"][0], emb["f32"][1], emb["f32"][2]),
        "qant": emb["qant"],
        "mixed_f32_index": (emb["qant"][0], emb["f32"][1], emb["f32"][2]),
        "qant_bf16_softmax": emb["qant_bf16softmax"],
    }
    S_ref = score_matrix(*variants["float32"], owner)
    top_ref, top1_ref = top10(S_ref), S_ref.argmax(1)
    out = {}
    for name, (eq, ed, eo) in variants.items():
        S = score_matrix(eq, ed, eo, owner)
        m = rank_metrics(S, owner)
        t = top10(S)
        m["top1_agreement"] = float((S.argmax(1) == top1_ref).mean())
        m["top10_overlap"] = float(np.mean([len(set(a) & set(b)) / 10 for a, b in zip(t, top_ref)]))
        out[name] = m
    for mode in ("qant", "qant_bf16softmax"):
        c = [float((emb["f32"][j] * emb[mode][j]).sum(1).mean()) for j in range(3)]
        cmin = min(float((emb["f32"][j] * emb[mode][j]).sum(1).min()) for j in range(3))
        out["qant" if mode == "qant" else "qant_bf16_softmax"]["mean_cosine_to_f32"] = float(np.mean(c))
        out["qant" if mode == "qant" else "qant_bf16_softmax"]["min_cosine_to_f32"] = cmin
    # how close the decisive margins are: top-1 minus top-2 score under float32
    srt = np.sort(S_ref, axis=1)
    out["float32"]["median_top1_margin"] = float(np.median(srt[:, -1] - srt[:, -2]))
    return out


def dynamic_mac_share(lengths):
    st = sum(L * (4 * D * D + 2 * D * FF) for L in lengths) * LAYERS
    dy = sum(2 * L * L * D for L in lengths) * LAYERS
    return dy / (st + dy)


def job(seed):
    paras = load_corpus()
    train_idx, test_idx, queries, index_docs = build_task(paras)
    vocab = Vocab(paras, train_idx)
    p = init_params(seed)
    untrained = evaluate(p, vocab, queries, index_docs)["float32"]
    losses = train(p, paras, train_idx, vocab, seed)
    res = evaluate(p, vocab, queries, index_docs)
    return {"seed": seed, "params": n_params(p), "loss_first_last": [losses[0], losses[-1]],
            "untrained_float32": untrained, **res}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processes", type=int, default=3)
    args = ap.parse_args()
    paras = load_corpus()
    train_idx, test_idx, queries, index_docs = build_task(paras)
    S, owner = bm25_scores(index_docs, queries)
    bm25 = rank_metrics(S, owner)
    vocab = Vocab(paras, train_idx)
    lengths = [len(vocab.encode(d)) for d in index_docs]
    with mp.get_context("fork").Pool(args.processes) as pool:
        rows = pool.map(job, (11, 22, 33))
    keys = ("float32", "qant", "mixed_f32_index", "qant_bf16_softmax", "untrained_float32")
    summary = {k: {m: statistics.fmean(r[k][m] for r in rows) for m in rows[0][k]} for k in keys}
    res = {"experiment": "CL-73", "corpus_paragraphs": len(paras), "train_paragraphs": len(train_idx),
           "test_paragraphs": len(test_idx), "queries": len(queries), "chance_recall_at_1": 1 / len(paras),
           "bm25": bm25, "dynamic_mac_share_index_docs": dynamic_mac_share(lengths),
           "mean_doc_tokens": statistics.fmean(lengths), "rows": rows, "summary": summary,
           "numpy": np.__version__}
    (HERE / "results.json").write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps({"bm25": bm25, "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
