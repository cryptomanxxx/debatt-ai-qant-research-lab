"""CL-51: stabilised local hidden rules and hidden-layer kWTA sparsity (Tier 0).

Pure-Python, offline, synthetic data. See README.md and PREDICTIONS.md.

Run from the repository root:
    python -m experiments.cl51_local_rules_and_hidden_sparsity.run
"""
from __future__ import annotations

import json
import math
import random
import statistics
import time
from pathlib import Path

from experiments.cl50_baselines_and_factorial.run import accuracy, or_rule_accuracy, summarize
from integrations.continual_learning.cl2_local_mlp import LocalHebbianMLP
from integrations.continual_learning.cl2_mlp_baseline import TinyDenseMLP
from integrations.continual_learning.cl2_shared_rule_candidate import (
    VERSION as DATA_VERSION,
    generate_shared_rule_candidate,
    shared_rule_dataset_sha256,
)

HERE = Path(__file__).resolve().parent
SEEDS = tuple(range(101, 131))
EPOCHS = 10
INPUTS, CLASSES = 16, 2
LR, LOCAL_LR = 0.05, 0.01


class GenMLP:
    """16 -> H -> 2 tanh MLP with sparse input traversal and optional kWTA.

    rule (hidden layer):
      backprop    - exact gradient (readout uses the same softmax gradient)
      hebb        - w += lr*h (active inputs), b += lr*h   [= LocalHebbianMLP]
      hebb_nobias - as hebb, biases not updated
      oja         - w += lr*h*(1 - h*w) on active inputs, no bias update
      competitive - kWTA winners move toward the input: w += lr*(x - w), all inputs
      dfa         - direct feedback alignment: fixed random B replaces W2^T
    Every non-backprop rule uses the local_hybrid readout: w2 += lr*(y - p)*h.
    k_hidden < H keeps only the k largest pre-activations (others output 0).
    """

    def __init__(self, seed: int, hidden: int = 8, rule: str = "backprop", k_hidden: int | None = None):
        rng = random.Random(seed)  # same draw order as TinyDenseMLP
        self.H, self.rule = hidden, rule
        self.k = hidden if k_hidden is None else k_hidden
        self.w1 = [[rng.uniform(-0.2, 0.2) for _ in range(INPUTS)] for _ in range(hidden)]
        self.b1 = [0.0] * hidden
        self.w2 = [[rng.uniform(-0.2, 0.2) for _ in range(hidden)] for _ in range(CLASSES)]
        self.b2 = [0.0] * CLASSES
        frng = random.Random(seed + 1_000_000)
        self.B = [[frng.uniform(-0.2, 0.2) for _ in range(hidden)] for _ in range(CLASSES)]
        self.writes = 0  # W1 + W2 weight writes (biases excluded)
        self.fwd_terms = 0  # W1 + W2 forward weight terms in training steps

    def _forward(self, x, count=False):
        active = [i for i, v in enumerate(x) if v]
        pre = [self.b1[j] + sum(self.w1[j][i] for i in active) for j in range(self.H)]
        if self.k < self.H:
            winners = sorted(sorted(range(self.H), key=lambda j: -pre[j])[:self.k])
        else:
            winners = list(range(self.H))
        hidden = [0.0] * self.H
        for j in winners:
            hidden[j] = math.tanh(pre[j])
        logits = [self.b2[c] + sum(self.w2[c][j] * hidden[j] for j in winners) for c in range(CLASSES)]
        if count:
            self.fwd_terms += self.H * len(active) + CLASSES * len(winners)
        return active, winners, hidden, logits

    def predict(self, x):
        logits = self._forward(x)[3]
        return max(range(CLASSES), key=lambda c: (logits[c], -c))

    def step(self, x, label):
        active, winners, h, logits = self._forward(x, count=True)
        m = max(logits)
        e = [math.exp(v - m) for v in logits]
        p = [v / sum(e) for v in e]
        err = [int(c == label) - p[c] for c in range(CLASSES)]  # = -dLoss/dlogit
        rule = self.rule
        if rule in ("backprop", "dfa"):
            fb = self.w2 if rule == "backprop" else self.B
            delta = {j: (1 - h[j] * h[j]) * sum(fb[c][j] * err[c] for c in range(CLASSES)) for j in winners}
            if rule == "backprop":
                # TinyDenseMLP order: W1 then W2, gradients from pre-update W2.
                for j in winners:
                    for i in active:
                        self.w1[j][i] += LR * delta[j]
                    self.b1[j] += LR * delta[j]
                for c in range(CLASSES):
                    for j in winners:
                        self.w2[c][j] += LR * (err[c] * h[j])  # same rounding order as TinyDenseMLP
                    self.b2[c] += LR * err[c]
                self.writes += len(winners) * len(active) + CLASSES * len(winners)
                return
        # Local-rule readout (LocalHebbianMLP order: readout first, then hidden).
        for c in range(CLASSES):
            for j in winners:
                self.w2[c][j] += LR * err[c] * h[j]
            self.b2[c] += LR * err[c]
        self.writes += CLASSES * len(winners)
        if rule == "dfa":
            for j in winners:
                for i in active:
                    self.w1[j][i] += LR * delta[j]
                self.b1[j] += LR * delta[j]
            self.writes += len(winners) * len(active)
        elif rule in ("hebb", "hebb_nobias"):
            for j in range(self.H):
                for i in active:
                    self.w1[j][i] += LOCAL_LR * h[j]
                if rule == "hebb":
                    self.b1[j] += LOCAL_LR * h[j]
            self.writes += self.H * len(active)
        elif rule == "oja":
            for j in range(self.H):
                for i in active:
                    self.w1[j][i] += LOCAL_LR * h[j] * (1 - h[j] * self.w1[j][i])
            self.writes += self.H * len(active)
        elif rule == "competitive":
            for j in winners:
                for i in range(INPUTS):
                    self.w1[j][i] += LOCAL_LR * (x[i] - self.w1[j][i])
            self.writes += len(winners) * INPUTS
        else:
            raise ValueError(rule)


def weights(m) -> list[float]:
    return [v for r in m.w1 for v in r] + list(m.b1) + [v for r in m.w2 for v in r] + list(m.b2)


def train(model, splits, epochs=EPOCHS):
    matrix = []
    for split in splits:
        for _ in range(epochs):
            for e in split.train:
                model.step(e.features, e.label)
        matrix.append([accuracy(model, s.test) for s in splits])
    return matrix


def score(matrix):
    final = matrix[-1]
    forgetting = [max(matrix[r][t] for r in range(t, 4)) - final[t] for t in range(3)]
    return sum(final) / 4, final, sum(forgetting) / 3


PART_A = {
    "backprop": dict(rule="backprop"),
    "hebb": dict(rule="hebb"),
    "hebb_nobias": dict(rule="hebb_nobias"),
    "oja": dict(rule="oja"),
    "competitive_k2": dict(rule="competitive", k_hidden=2),
    "dfa": dict(rule="dfa"),
}
PART_B = {f"backprop_H{H}_k{k}": dict(rule="backprop", hidden=H, k_hidden=k)
          for H, ks in ((8, (1, 2, 4, 8)), (32, (2, 4, 8, 32))) for k in ks}


def run(seeds=SEEDS, epochs=EPOCHS, parts=(PART_A, PART_B)):
    rng = random.Random(0)
    data = {s: generate_shared_rule_candidate(s) for s in seeds}
    out = {"experiment": "CL-51", "data_version": DATA_VERSION, "seeds": list(seeds),
           "epochs_per_task": epochs,
           "dataset_sha256": {str(s): shared_rule_dataset_sha256(data[s]) for s in seeds},
           "or_rule": statistics.fmean(statistics.fmean(or_rule_accuracy(t.test) for t in data[s])
                                       for s in seeds),
           "arms": {}}
    timing, finals = {}, {}
    # Correctness: GenMLP must reproduce the existing CL-2 classes bit for bit.
    ident = {"backprop==TinyDenseMLP": True, "hebb==LocalHebbianMLP": True}
    for s in seeds:
        for name, ref_cls, rule in (("backprop==TinyDenseMLP", TinyDenseMLP, "backprop"),
                                    ("hebb==LocalHebbianMLP", LocalHebbianMLP, "hebb")):
            a, b = GenMLP(s, rule=rule), ref_cls(seed=s)
            if train(a, data[s], epochs) != train(b, data[s], epochs) or weights(a) != weights(b):
                ident[name] = False
    out["bitwise_identity_with_cl2_classes"] = ident
    for part in parts:
        for arm, cfg in part.items():
            start = time.perf_counter()
            runs = []
            for s in seeds:
                m = GenMLP(s, **cfg)
                matrix = train(m, data[s], epochs)
                mean, final, forget = score(matrix)
                test = [e.features for t in data[s] for e in t.test]
                runs.append({"final_mean": mean, "final": final, "forgetting": forget,
                             "one_class": len({m.predict(x) for x in test}) == 1,
                             "max_abs_w1": max(abs(v) for r in m.w1 for v in r),
                             "fwd": m.fwd_terms, "writes": m.writes})
            timing[arm] = time.perf_counter() - start
            finals[arm] = [r["final_mean"] for r in runs]
            out["arms"][arm] = {
                "config": {"hidden": cfg.get("hidden", 8), "k_hidden": cfg.get("k_hidden", cfg.get("hidden", 8)),
                           "rule": cfg["rule"]},
                "final_mean": summarize(finals[arm], rng),
                "final_per_task_mean": [statistics.fmean(r["final"][t] for r in runs) for t in range(4)],
                "mean_forgetting_T1_T3": statistics.fmean(r["forgetting"] for r in runs),
                "seeds_ge_0.65": sum(r["final_mean"] >= 0.65 for r in runs),
                "seeds_predicting_one_class": sum(r["one_class"] for r in runs),
                "max_abs_w1_mean": statistics.fmean(r["max_abs_w1"] for r in runs),
                "train_forward_weight_terms_mean": statistics.fmean(r["fwd"] for r in runs),
                "train_weight_writes_mean": statistics.fmean(r["writes"] for r in runs),
            }
    paired = {}
    for a, b in (("dfa", "backprop"), ("oja", "hebb"), ("competitive_k2", "hebb"), ("dfa", "hebb"),
                 ("backprop_H8_k4", "backprop_H8_k8"), ("backprop_H32_k4", "backprop_H8_k8")):
        if a in finals and b in finals:
            paired[f"{a} - {b}"] = summarize([x - y for x, y in zip(finals[a], finals[b])], rng)
    out["paired_final_mean_differences"] = paired
    return out, timing


def main():
    results, timing = run()
    (HERE / "results.json").write_text(json.dumps(results, indent=1, sort_keys=True) + "\n")
    (HERE / "timing.json").write_text(json.dumps(timing, indent=1, sort_keys=True) + "\n")
    print("identity:", results["bitwise_identity_with_cl2_classes"], " OR rule:", round(results["or_rule"], 3))
    for arm, r in results["arms"].items():
        m = r["final_mean"]
        print(f"{arm:18s} {m['mean']:.3f} [{m['ci95'][0]:.3f},{m['ci95'][1]:.3f}] "
              f"tasks {[round(v, 3) for v in r['final_per_task_mean']]} forget {r['mean_forgetting_T1_T3']:.3f} "
              f">=.65 {r['seeds_ge_0.65']} 1cls {r['seeds_predicting_one_class']} |w1| {r['max_abs_w1_mean']:.2f} "
              f"fwd {r['train_forward_weight_terms_mean']:.0f} writes {r['train_weight_writes_mean']:.0f}")
    for k, d in results["paired_final_mean_differences"].items():
        print(f"  {k:36s} {d['mean']:+.3f} [{d['ci95'][0]:+.3f},{d['ci95'][1]:+.3f}]")
    print("timing s:", {k: round(v, 1) for k, v in timing.items()})


if __name__ == "__main__":
    main()
