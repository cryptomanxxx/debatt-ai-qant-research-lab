"""CL-72: which applications could benefit from Q.ANT's linear primitive? (Tier 0)

Part A (analytic): per-workload accounting of where the MACs are (static-weight matrix
products vs activation x activation products), fan-in, arithmetic intensity and host-side
work, for standard published shapes. A calculation, not an experiment.
Part B (measured, simulator): a LeNet-style CNN on Fashion-MNIST, convolutions as im2col
matrix products, evaluated on the simulated Q.ANT path (CL-70 model of linear_fprop).
See PREDICTIONS.md.

Run from the repository root:
    python -m experiments.cl72_application_mapping.run [--processes N]
"""
from __future__ import annotations

import experiments.cl58_qant_quantization_aware_training.run as cl58  # sets single-threaded BLAS first

import argparse
import json
import multiprocessing as mp
import statistics
from pathlib import Path

import numpy as np

bf16 = cl58.bf16
HERE = Path(__file__).resolve().parent
B2 = 2  # bytes per bf16 value


# ------------------------------------------------------------------ Part A: analytic accounting

class Workload:
    def __init__(self, name, note):
        self.name, self.note, self.layers = name, note, []
        self.host_ops = 0        # non-matrix elementwise operations (activations, norms, softmax, residual adds)
        self.extra_bytes = 0     # memory traffic not tied to a matrix product (embeddings, KV cache)

    def mm(self, macs_per_out, n_out, fan_in, n_in, static, weights=0):
        """A matrix product producing n_out outputs, each a dot product of length fan_in."""
        self.layers.append({"macs": macs_per_out * n_out, "fan_in": fan_in, "io_bytes": B2 * (n_in + n_out),
                            "static": static, "weights": weights})

    def linear(self, n_tokens, d_in, d_out, act=True):
        self.mm(d_in, n_tokens * d_out, d_in, n_tokens * d_in, True, d_in * d_out + d_out)
        if act:
            self.host_ops += n_tokens * d_out

    def conv(self, h_out, w_out, c_in, c_out, k, h_in, w_in, groups=1, act=True):
        fan = k * k * c_in // groups
        self.mm(fan, h_out * w_out * c_out, fan, h_in * w_in * c_in, True, fan * c_out + c_out)
        if act:
            self.host_ops += h_out * w_out * c_out

    def summary(self):
        macs = sum(l["macs"] for l in self.layers)
        static = sum(l["macs"] for l in self.layers if l["static"])
        io = sum(l["io_bytes"] for l in self.layers) + self.extra_bytes
        return {"workload": self.name, "note": self.note, "total_macs": macs,
                "static_mac_share": static / macs, "dynamic_mac_share": 1 - static / macs,
                "mac_weighted_mean_fan_in": sum(l["macs"] * l["fan_in"] for l in self.layers) / macs,
                "min_fan_in": min(l["fan_in"] for l in self.layers),
                "arithmetic_intensity_macs_per_byte": macs / io,
                "extra_bytes": self.extra_bytes, "weights_mb_bf16": B2 * sum(l["weights"] for l in self.layers) / 1e6,
                "host_ops_per_mac": self.host_ops / macs}


def mlp():
    w = Workload("MLP 784-256-256-10", "dense classifier")
    w.linear(1, 784, 256); w.linear(1, 256, 256); w.linear(1, 256, 10, act=False)
    return w


def lenet():
    w = Workload("LeNet-style CNN (Part B)", "28x28 input")
    w.conv(24, 24, 1, 16, 5, 28, 28); w.host_ops += 12 * 12 * 16
    w.conv(8, 8, 16, 32, 5, 12, 12); w.host_ops += 4 * 4 * 32
    w.linear(1, 512, 10, act=False)
    return w


def resnet18():
    w = Workload("ResNet-18", "224x224 image")
    w.conv(112, 112, 3, 64, 7, 224, 224); w.host_ops += 56 * 56 * 64
    res, c = 56, 64
    for stage, c_out in enumerate((64, 128, 256, 512)):
        for block in range(2):
            stride = 2 if (stage > 0 and block == 0) else 1
            r_out = res // stride
            w.conv(r_out, r_out, c, c_out, 3, res, res)
            w.conv(r_out, r_out, c_out, c_out, 3, r_out, r_out)
            if stride == 2 or c != c_out:
                w.conv(r_out, r_out, c, c_out, 1, res, res, act=False)
            w.host_ops += r_out * r_out * c_out  # residual add
            res, c = r_out, c_out
    w.host_ops += 7 * 7 * 512
    w.linear(1, 512, 1000, act=False)
    return w


def mobilenet_v2():
    w = Workload("MobileNetV2", "224x224 image; depthwise convolutions")
    w.conv(112, 112, 3, 32, 3, 224, 224)
    res, c = 112, 32
    for t, c_out, n, s in ((1, 16, 1, 1), (6, 24, 2, 2), (6, 32, 3, 2), (6, 64, 4, 2), (6, 96, 3, 1), (6, 160, 3, 2), (6, 320, 1, 1)):
        for i in range(n):
            stride = s if i == 0 else 1
            hidden = c * t
            if t != 1:
                w.conv(res, res, c, hidden, 1, res, res)
            r_out = res // stride
            w.conv(r_out, r_out, hidden, hidden, 3, res, res, groups=hidden)
            w.conv(r_out, r_out, hidden, c_out, 1, r_out, r_out, act=False)
            if stride == 1 and c == c_out:
                w.host_ops += r_out * r_out * c_out
            res, c = r_out, c_out
    w.conv(7, 7, 320, 1280, 1, 7, 7); w.host_ops += 7 * 7 * 1280
    w.linear(1, 1280, 1000, act=False)
    return w


def bert(seq):
    w = Workload(f"BERT-base encoder, sequence {seq}", "12 layers, d=768, 12 heads")
    d, h, ff = 768, 12, 3072
    for _ in range(12):
        w.linear(seq, d, 3 * d, act=False)
        dh = d // h
        w.mm(dh, h * seq * seq, dh, 2 * seq * d, False)        # Q K^T (activation x activation)
        w.host_ops += 5 * h * seq * seq                          # softmax
        w.mm(seq, h * seq * dh, seq, h * seq * seq + seq * d, False)  # A V
        w.linear(seq, d, d, act=False)
        w.host_ops += 2 * 8 * seq * d                            # two residual + layer norms
        w.linear(seq, d, ff)                                     # GELU counted as host op
        w.linear(seq, ff, d, act=False)
    w.extra_bytes += B2 * seq * d                                # token embedding lookup
    return w


def gpt2_decode(ctx):
    w = Workload(f"GPT-2-small decode, context {ctx}", "one new token; KV cache read")
    d, h, vocab = 768, 12, 50257
    for _ in range(12):
        w.linear(1, d, 3 * d, act=False)
        dh = d // h
        w.mm(dh, h * ctx, dh, d + ctx * d, False)               # q . K^T over the cache
        w.host_ops += 5 * h * ctx
        w.mm(ctx, h * dh, ctx, h * ctx + ctx * d, False)        # A V over the cache
        w.linear(1, d, d, act=False)
        w.host_ops += 2 * 8 * d
        w.linear(1, d, 4 * d)
        w.linear(1, 4 * d, d, act=False)
    w.linear(1, d, vocab, act=False)
    return w


def dlrm():
    w = Workload("DLRM-style recommender", "26 embedding tables x 64; MLPs")
    w.linear(1, 13, 512); w.linear(1, 512, 256); w.linear(1, 256, 64)
    w.extra_bytes += B2 * 26 * 64                                # embedding lookups
    n = 27
    w.mm(64, n * (n - 1) // 2, 64, n * 64, False)                # pairwise dot-product interaction
    w.linear(1, 64 + n * (n - 1) // 2, 512); w.linear(1, 512, 256); w.linear(1, 256, 1, act=False)
    return w


def lstm():
    w = Workload("2-layer LSTM, hidden 512", "one time step; sequential dependency")
    for _ in range(2):
        w.linear(1, 1024, 4 * 512, act=False)
        w.host_ops += 4 * 512 + 5 * 512
    return w


def part_a():
    return [f().summary() for f in (mlp, lenet, resnet18, mobilenet_v2, lambda: bert(128), lambda: bert(2048),
                                    lambda: gpt2_decode(1024), dlrm, lstm)]


# ------------------------------------------------------------------ Part B: CNN on the simulated Q.ANT path

def im2col(x, k):
    """x (B, H, W, C) -> (B, Ho, Wo, k*k*C), offsets (di, dj) outer, channels inner."""
    B, H, W, C = x.shape
    ho, wo = H - k + 1, W - k + 1
    return np.concatenate([x[:, i:i + ho, j:j + wo, :] for i in range(k) for j in range(k)], axis=3)


def col2im(g, k, shape):
    B, H, W, C = shape
    ho, wo = H - k + 1, W - k + 1
    dx = np.zeros(shape, np.float32)
    for n, (i, j) in enumerate((i, j) for i in range(k) for j in range(k)):
        dx[:, i:i + ho, j:j + wo, :] += g[..., n * C:(n + 1) * C]
    return dx


def pool(x):
    B, H, W, C = x.shape
    return x.reshape(B, H // 2, 2, W // 2, 2, C).mean((2, 4))


def unpool(g):
    return np.repeat(np.repeat(g, 2, axis=1), 2, axis=2) / 4


class CNN:
    SHAPES = (("c1", 5, 1, 16), ("c2", 5, 16, 32), ("fc", None, 512, 10))

    def __init__(self, seed):
        rng = np.random.default_rng(seed)
        self.p = {}
        for name, k, cin, cout in self.SHAPES:
            fan = (k * k * cin) if k else cin
            b = 1 / np.sqrt(fan)
            self.p[name + "_W"] = rng.uniform(-b, b, (cout, fan)).astype(np.float32)
            self.p[name + "_b"] = rng.uniform(-b, b, cout).astype(np.float32)
        self.m = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.t = 0

    def forward(self, X):
        x0 = X.reshape(-1, 28, 28, 1)
        c1 = im2col(x0, 5)
        z1 = c1 @ self.p["c1_W"].T + self.p["c1_b"]
        a1 = np.maximum(z1, 0)
        p1 = pool(a1)
        c2 = im2col(p1, 5)
        z2 = c2 @ self.p["c2_W"].T + self.p["c2_b"]
        a2 = np.maximum(z2, 0)
        p2 = pool(a2).reshape(len(X), -1)
        logits = p2 @ self.p["fc_W"].T + self.p["fc_b"]
        return logits, (x0, c1, z1, p1, c2, z2, p2)

    def train_batch(self, X, y):
        logits, (x0, c1, z1, p1, c2, z2, p2) = self.forward(X)
        z = logits - logits.max(1, keepdims=True)
        pr = np.exp(z)
        pr /= pr.sum(1, keepdims=True)
        pr[np.arange(len(y)), y] -= 1
        g = (pr / len(y)).astype(np.float32)
        grads = {"fc_W": g.T @ p2, "fc_b": g.sum(0)}
        gp2 = (g @ self.p["fc_W"]).reshape(len(X), 4, 4, 32)
        gz2 = unpool(gp2) * (z2 > 0)
        grads["c2_W"] = gz2.reshape(-1, 32).T @ c2.reshape(-1, c2.shape[-1])
        grads["c2_b"] = gz2.sum((0, 1, 2))
        gc2 = gz2 @ self.p["c2_W"]
        gp1 = col2im(gc2, 5, p1.shape)
        gz1 = unpool(gp1) * (z1 > 0)
        grads["c1_W"] = gz1.reshape(-1, 16).T @ c1.reshape(-1, c1.shape[-1])
        grads["c1_b"] = gz1.sum((0, 1, 2))
        self.t += 1
        for k, gr in grads.items():
            self.m[k] = 0.9 * self.m[k] + 0.1 * gr
            self.v[k] = 0.999 * self.v[k] + 0.001 * gr * gr
            mh = self.m[k] / (1 - 0.9 ** self.t)
            vh = self.v[k] / (1 - 0.999 ** self.t)
            self.p[k] -= (1e-3 * mh / (np.sqrt(vh) + 1e-8)).astype(np.float32)

    def qant_logits(self, X):
        """Every matrix product through simulated linear_fprop (bf16 in/out, float32 accumulation)."""
        def lin(a, name):
            o = bf16(bf16(a) @ bf16(self.p[name + "_W"]).T)
            return bf16(o + bf16(self.p[name + "_b"]))
        x0 = bf16(X.reshape(-1, 28, 28, 1))
        a1 = np.maximum(lin(im2col(x0, 5), "c1"), 0)
        p1 = bf16(pool(a1))                       # host pooling in float32, recast
        a2 = np.maximum(lin(im2col(p1, 5), "c2"), 0)
        p2 = bf16(pool(a2).reshape(len(X), -1))
        return lin(p2, "fc")


def cnn_job(args):
    seed, data_dir = args
    x_tr, y_tr, x_te, y_te = cl58.load_fashion_mnist(Path(data_dir))
    model = CNN(seed)
    rng = np.random.default_rng(seed + 1)
    for _ in range(3):
        order = rng.permutation(len(x_tr))
        for s in range(0, len(order), 128):
            idx = order[s:s + 128]
            model.train_batch(x_tr[idx], y_tr[idx])
    ref = np.concatenate([model.forward(x_te[s:s + 500])[0].argmax(1) for s in range(0, len(x_te), 500)])
    q = np.concatenate([model.qant_logits(x_te[s:s + 500]).argmax(1) for s in range(0, len(x_te), 500)])
    return {"seed": seed, "float_accuracy": float((ref == y_te).mean()), "qant_accuracy": float((q == y_te).mean()),
            "gap_pp": 100 * float((q == y_te).mean() - (ref == y_te).mean()), "disagreements": int((ref != q).sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=cl58.HERE / "data")
    ap.add_argument("--processes", type=int, default=3)
    args = ap.parse_args()
    a = part_a()
    cl58.load_fashion_mnist(args.data_dir)
    with mp.get_context("fork").Pool(args.processes) as pool_:
        b = pool_.map(cnn_job, [(s, str(args.data_dir)) for s in (11, 22, 33)])
    bsum = {k: statistics.fmean(r[k] for r in b) for k in b[0] if k != "seed"}
    res = {"experiment": "CL-72", "part_a": a, "part_b_rows": b, "part_b_summary": bsum, "numpy": np.__version__}
    (HERE / "results.json").write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")
    print(f"{'workload':38s} {'GMAC':>8s} {'static':>7s} {'dyn':>6s} {'fan-in':>7s} {'min':>5s} {'MAC/B':>7s} {'W MB':>7s} {'host/MAC':>8s}")
    for s in a:
        print(f"{s['workload']:38s} {s['total_macs']/1e9:8.4f} {s['static_mac_share']:7.3f} {s['dynamic_mac_share']:6.3f}"
              f" {s['mac_weighted_mean_fan_in']:7.0f} {s['min_fan_in']:5d} {s['arithmetic_intensity_macs_per_byte']:7.1f}"
              f" {s['weights_mb_bf16']:7.1f} {s['host_ops_per_mac']:8.4f}")
    print("Part B CNN:", {k: round(v, 4) for k, v in bsum.items()}, [r["disagreements"] for r in b])


if __name__ == "__main__":
    main()
