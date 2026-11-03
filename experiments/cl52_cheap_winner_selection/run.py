"""CL-52: cheap selection of active hidden units (Tier 0).

Can the k winners of a kWTA hidden layer be chosen without computing every
pre-activation? Pure-Python, offline, synthetic data. See README.md and
PREDICTIONS.md.

Run from the repository root:
    python -m experiments.cl52_cheap_winner_selection.run
"""
from __future__ import annotations

import json
import math
import random
import statistics
import time
from pathlib import Path

from experiments.cl50_baselines_and_factorial.run import accuracy, summarize
from experiments.cl51_local_rules_and_hidden_sparsity.run import GenMLP, train as train51, weights
from integrations.continual_learning.cl2_shared_rule_candidate import (
    VERSION as DATA_VERSION,
    generate_shared_rule_candidate,
    shared_rule_dataset_sha256,
)

HERE = Path(__file__).resolve().parent
SEEDS = tuple(range(101, 131))
EPOCHS = 10
INPUTS, CLASSES, LR = 16, 2, 0.05
REFRESH = 128  # steps between selector refreshes (low-rank / hash tables)


class SelMLP:
    """16 -> H -> 2 tanh MLP, backprop, k winners chosen by a selector.

    Work counters (per call, accumulated separately for training and eval):
      select - screening/hashing operations, incl. amortised refresh cost
      pre    - W1 terms computed exactly (units evaluated x active inputs)
      out    - W2 terms (classes x winners)
    """

    def __init__(self, seed, hidden=32, k=4, selector="exact", r=2, c=8, K=3, L=2):
        rng = random.Random(seed)  # same draw order as TinyDenseMLP / GenMLP
        self.H, self.k, self.selector = hidden, k, selector
        self.w1 = [[rng.uniform(-0.2, 0.2) for _ in range(INPUTS)] for _ in range(hidden)]
        self.b1 = [0.0] * hidden
        self.w2 = [[rng.uniform(-0.2, 0.2) for _ in range(hidden)] for _ in range(CLASSES)]
        self.b2 = [0.0] * CLASSES
        self.rng = random.Random(seed + 2_000_000)
        self.r, self.c, self.K, self.L = r, c, K, L
        self.steps = 0
        self.work = {"select": 0, "pre": 0, "out": 0}
        self.writes = 0
        if selector == "lowrank":
            vrng = random.Random(seed + 3_000_000)
            self.V = [[vrng.gauss(0, 1) for _ in range(r)] for _ in range(INPUTS)]
            self._refresh_lowrank(iters=10)
        elif selector == "simhash":
            hrng = random.Random(seed + 4_000_000)
            # K signed projections per table over [x; 1] (17 dims), entries +-1.
            self.P = [[[hrng.choice((-1, 1)) for _ in range(INPUTS + 1)] for _ in range(K)] for _ in range(L)]
            self._refresh_hash()

    # ------------------------------------------------------------ selectors
    def _refresh_lowrank(self, iters=2):
        """Warm-started power iteration for the top-r right singular vectors of W1."""
        H, r = self.H, self.r
        V = self.V
        for _ in range(iters):
            U = [[sum(self.w1[j][i] * V[i][q] for i in range(INPUTS)) for q in range(r)] for j in range(H)]
            V = [[sum(self.w1[j][i] * U[j][q] for j in range(H)) for q in range(r)] for i in range(INPUTS)]
            for q in range(r):  # Gram-Schmidt
                for p in range(q):
                    d = sum(V[i][q] * V[i][p] for i in range(INPUTS))
                    for i in range(INPUTS):
                        V[i][q] -= d * V[i][p]
                n = math.sqrt(sum(V[i][q] ** 2 for i in range(INPUTS))) or 1.0
                for i in range(INPUTS):
                    V[i][q] /= n
        self.V = V
        self.A = [[sum(self.w1[j][i] * V[i][q] for i in range(INPUTS)) for q in range(r)] for j in range(H)]
        return iters * 2 * H * INPUTS * r + H * INPUTS * r + iters * r * r * INPUTS

    def _refresh_hash(self):
        self.unit_codes = [[tuple(sum(p[i] * w for i, w in enumerate(self.w1[j] + [self.b1[j]])) > 0
                                  for p in self.P[t]) for j in range(self.H)] for t in range(self.L)]
        return self.H * self.L * self.K * (INPUTS + 1)

    def _select(self, active, training):
        H, k = self.H, self.k
        select = 0
        if training and self.selector in ("lowrank", "simhash") and self.steps and self.steps % REFRESH == 0:
            select += self._refresh_lowrank() if self.selector == "lowrank" else self._refresh_hash()
        if self.selector == "exact":
            cand = list(range(H))
        elif self.selector == "static":
            cand = list(range(k))
        elif self.selector == "random":
            cand = sorted(self.rng.sample(range(H), k))
        elif self.selector == "lowrank":
            z = [sum(self.V[i][q] for i in active) for q in range(self.r)]
            score = [self.b1[j] + sum(self.A[j][q] * z[q] for q in range(self.r)) for j in range(H)]
            select += self.r * len(active) + H * self.r
            cand = sorted(range(H), key=lambda j: -score[j])[:self.c]
        elif self.selector == "simhash":
            cand_set = set()
            for t in range(self.L):
                code = tuple(sum(self.P[t][b][i] for i in active) + self.P[t][b][INPUTS] > 0 for b in range(self.K))
                cand_set.update(j for j in range(H) if self.unit_codes[t][j] == code)
            select += self.L * self.K * (len(active) + 1)
            cand = sorted(cand_set)
            if len(cand) < k:  # pad as SLIDE does, with random units
                rest = [j for j in range(H) if j not in cand_set]
                cand = sorted(cand + self.rng.sample(rest, k - len(cand)))
        else:
            raise ValueError(self.selector)
        pre = {j: self.b1[j] + sum(self.w1[j][i] for i in active) for j in cand}
        winners = sorted(sorted(cand, key=lambda j: -pre[j])[:k])
        return winners, pre, select, len(cand) * len(active)

    def _forward(self, x, training=False, work=None):
        active = [i for i, v in enumerate(x) if v]
        winners, pre, select, pre_terms = self._select(active, training)
        h = {j: math.tanh(pre[j]) for j in winners}
        logits = [self.b2[c] + sum(self.w2[c][j] * h[j] for j in winners) for c in range(CLASSES)]
        if work is not None:
            work["select"] += select
            work["pre"] += pre_terms
            work["out"] += CLASSES * len(winners)
        return active, winners, h, logits

    def predict(self, x, work=None):
        logits = self._forward(x, work=work)[3]
        return max(range(CLASSES), key=lambda c: (logits[c], -c))

    def step(self, x, label):
        active, winners, h, logits = self._forward(x, training=True, work=self.work)
        m = max(logits)
        e = [math.exp(v - m) for v in logits]
        p = [v / sum(e) for v in e]
        err = [int(c == label) - p[c] for c in range(CLASSES)]
        delta = {j: (1 - h[j] * h[j]) * sum(self.w2[c][j] * err[c] for c in range(CLASSES)) for j in winners}
        for j in winners:
            for i in active:
                self.w1[j][i] += LR * delta[j]
            self.b1[j] += LR * delta[j]
        for c in range(CLASSES):
            for j in winners:
                self.w2[c][j] += LR * (err[c] * h[j])
            self.b2[c] += LR * err[c]
        self.writes += len(winners) * len(active) + CLASSES * len(winners)
        self.steps += 1

    def exact_winners(self, x):
        active = [i for i, v in enumerate(x) if v]
        pre = [self.b1[j] + sum(self.w1[j][i] for i in active) for j in range(self.H)]
        return set(sorted(range(self.H), key=lambda j: -pre[j])[:self.k])


ARMS = {
    "exact": dict(selector="exact"),
    "dense_H8": dict(selector="exact", hidden=8, k=8),
    "static_first_k": dict(selector="static"),
    "random": dict(selector="random"),
    "lowrank_r2_c8": dict(selector="lowrank", r=2, c=8),
    "lowrank_r4_c8": dict(selector="lowrank", r=4, c=8),
    "simhash_K3_L2": dict(selector="simhash", K=3, L=2),
    "simhash_K2_L4": dict(selector="simhash", K=2, L=4),
}


def run(seeds=SEEDS, epochs=EPOCHS, arms=ARMS):
    rng = random.Random(0)
    data = {s: generate_shared_rule_candidate(s) for s in seeds}
    out = {"experiment": "CL-52", "data_version": DATA_VERSION, "seeds": list(seeds), "epochs_per_task": epochs,
           "refresh_every_steps": REFRESH,
           "dataset_sha256": {str(s): shared_rule_dataset_sha256(data[s]) for s in seeds}, "arms": {}}
    # Correctness: exact selector must equal CL-51 GenMLP bit for bit.
    ident = {}
    for name, cfg, gen in (("exact==GenMLP_H32_k4", ARMS["exact"], dict(hidden=32, k_hidden=4)),
                           ("dense_H8==GenMLP_H8", ARMS["dense_H8"], dict(hidden=8))):
        ok = True
        for s in seeds:
            a, b = SelMLP(s, **cfg), GenMLP(s, rule="backprop", **gen)
            if train51(a, data[s], epochs) != train51(b, data[s], epochs) or weights(a) != weights(b):
                ok = False
        ident[name] = ok
    out["bitwise_identity"] = ident
    timing, finals = {}, {}
    for arm, cfg in arms.items():
        start = time.perf_counter()
        runs = []
        for s in seeds:
            m = SelMLP(s, **cfg)
            matrix = train51(m, data[s], epochs)
            final = matrix[-1]
            test = [e.features for t in data[s] for e in t.test]
            ework = {"select": 0, "pre": 0, "out": 0}
            recall = 0.0
            for x in test:
                winners = m._forward(x, work=ework)[1]
                recall += len(set(winners) & m.exact_winners(x)) / m.k
            steps = m.steps
            runs.append({
                "final_mean": sum(final) / 4, "final": final,
                "forgetting": sum(max(matrix[r][t] for r in range(t, 4)) - final[t] for t in range(3)) / 3,
                "train_work_per_step": {key: v / steps for key, v in m.work.items()},
                "eval_work_per_input": {key: v / len(test) for key, v in ework.items()},
                "writes_per_step": m.writes / steps,
                "top_k_recall": recall / len(test),
            })
        timing[arm] = time.perf_counter() - start
        finals[arm] = [r["final_mean"] for r in runs]

        def mean_of(field, key=None):
            return statistics.fmean(r[field][key] if key else r[field] for r in runs)

        tw = {key: mean_of("train_work_per_step", key) for key in ("select", "pre", "out")}
        ew = {key: mean_of("eval_work_per_input", key) for key in ("select", "pre", "out")}
        out["arms"][arm] = {
            "config": cfg,
            "final_mean": summarize(finals[arm], rng),
            "final_per_task_mean": [statistics.fmean(r["final"][t] for r in runs) for t in range(4)],
            "mean_forgetting_T1_T3": mean_of("forgetting"),
            "train_work_per_step": tw, "train_work_per_step_total": sum(tw.values()),
            "eval_work_per_input": ew, "eval_work_per_input_total": sum(ew.values()),
            "writes_per_step": mean_of("writes_per_step"),
            "top_k_recall": mean_of("top_k_recall"),
        }
    ref = out["arms"]["exact"]["train_work_per_step_total"]
    for a in out["arms"].values():
        a["train_work_vs_exact"] = a["train_work_per_step_total"] / ref
    paired = {}
    for a in arms:
        for b in ("exact", "static_first_k"):
            if a != b:
                paired[f"{a} - {b}"] = summarize([x - y for x, y in zip(finals[a], finals[b])], rng)
    out["paired_final_mean_differences"] = paired
    return out, timing


def main():
    results, timing = run()
    (HERE / "results.json").write_text(json.dumps(results, indent=1, sort_keys=True) + "\n")
    (HERE / "timing.json").write_text(json.dumps(timing, indent=1, sort_keys=True) + "\n")
    print("identity:", results["bitwise_identity"])
    for arm, r in results["arms"].items():
        m = r["final_mean"]
        tw = r["train_work_per_step"]
        print(f"{arm:16s} acc {m['mean']:.3f} [{m['ci95'][0]:.3f},{m['ci95'][1]:.3f}] recall {r['top_k_recall']:.2f} "
              f"train/step {r['train_work_per_step_total']:6.1f} (sel {tw['select']:.1f} pre {tw['pre']:.1f} out {tw['out']:.1f}) "
              f"vs exact {r['train_work_vs_exact']:.2f} eval/input {r['eval_work_per_input_total']:6.1f} "
              f"writes {r['writes_per_step']:.1f} forget {r['mean_forgetting_T1_T3']:.3f}")
    for k, d in results["paired_final_mean_differences"].items():
        print(f"  {k:34s} {d['mean']:+.3f} [{d['ci95'][0]:+.3f},{d['ci95'][1]:+.3f}]")
    print("timing s:", {k: round(v, 1) for k, v in timing.items()})


if __name__ == "__main__":
    main()
