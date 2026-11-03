"""CL-50: baselines and dense/sparse x backprop/local comparison (Tier 0).

Pure-Python, offline, synthetic data, seconds of CPU. See README.md.

Run from the repository root:
    python -m experiments.cl50_baselines_and_factorial.run
"""
from __future__ import annotations

import json
import math
import random
import statistics
import sys
import time
from pathlib import Path

from integrations.continual_learning.cl2_local_mlp import LocalHebbianMLP
from integrations.continual_learning.cl2_mlp_baseline import (
    HIDDEN_SIZE,
    INPUT_SIZE,
    N_CLASSES,
    TinyDenseMLP,
)
from integrations.continual_learning.cl2_shared_index_forward import SharedIndexSparseForwardLocalMLP
from integrations.continual_learning.cl2_shared_rule_candidate import (
    VERSION as DATA_VERSION,
    generate_shared_rule_candidate,
    shared_rule_dataset_sha256,
)
from integrations.continual_learning.cl2_sparse_forward import SparseForwardLocalMLP

HERE = Path(__file__).resolve().parent
SEEDS = tuple(range(101, 131))  # 30 paired seeds; 101-105 are the CL-3X draft seeds
EPOCH_SETTINGS = (1, 10)        # 1 = CL-3X draft (one pass per task); 10 = exploratory
TASKS = ("T1", "T2", "T3", "T4")
BOOTSTRAP_RESAMPLES = 2000


# ---------------------------------------------------------------- extra arms

class SparseAwareSGD(TinyDenseMLP):
    """Backprop with sparse forward and W1 writes only for active inputs.

    For binary inputs the W1 gradient of an inactive input is exactly zero,
    so skipping those writes should leave every weight bit-identical to
    dense SGD. The run checks this instead of assuming it.
    """

    def _forward(self, x):
        active = [i for i, v in enumerate(x) if v]
        hidden = [math.tanh(self.b1[j] + sum(self.w1[j][i] for i in active))
                  for j in range(HIDDEN_SIZE)]
        logits = [self.b2[k] + sum(self.w2[k][j] * hidden[j] for j in range(HIDDEN_SIZE))
                  for k in range(N_CLASSES)]
        return hidden, logits

    def step(self, features, label):
        x = self._input(features)
        loss, grad = self.loss_and_gradients(x, label)
        active = [i for i, v in enumerate(x) if v]
        rate = self.learning_rate
        for j in range(HIDDEN_SIZE):
            for i in active:
                self.w1[j][i] -= rate * grad.w1[j][i]
            self.b1[j] -= rate * grad.b1[j]
        for k in range(N_CLASSES):
            for j in range(HIDDEN_SIZE):
                self.w2[k][j] -= rate * grad.w2[k][j]
            self.b2[k] -= rate * grad.b2[k]
        self.optimizer_steps += 1
        return loss


def readout_only(seed: int) -> LocalHebbianMLP:
    """Control: same model, hidden layer frozen at its random init."""
    model = LocalHebbianMLP(seed=seed)
    model.local_rate = 0.0  # w1/b1 updates become += 0.0 * h
    return model


class Logistic:
    """Linear baseline on the raw 16 inputs, same SGD learning rate."""

    def __init__(self, learning_rate: float = 0.05):
        self.w = [0.0] * INPUT_SIZE
        self.b = 0.0
        self.learning_rate = learning_rate

    def _z(self, x):
        return self.b + sum(self.w[i] for i, v in enumerate(x) if v)

    def predict(self, x):
        return int(self._z(x) > 0)

    def step(self, x, label):
        p = 1 / (1 + math.exp(-self._z(x)))
        g = p - label
        for i, v in enumerate(x):
            if v:
                self.w[i] -= self.learning_rate * g
        self.b -= self.learning_rate * g


ARMS = {
    # name: (factory, traversal, learning rule)
    "dense_sgd": (lambda s: TinyDenseMLP(seed=s), "dense", "backprop"),
    "sparse_sgd": (lambda s: SparseAwareSGD(seed=s), "sparse", "backprop"),
    "local_hybrid": (lambda s: LocalHebbianMLP(seed=s), "dense", "local"),
    "sparse_local_hybrid": (lambda s: SparseForwardLocalMLP(seed=s), "sparse", "local"),
    "sparse_shared_index": (lambda s: SharedIndexSparseForwardLocalMLP(seed=s), "sparse", "local"),
    "readout_only": (readout_only, "dense", "readout only"),
    "logistic": (lambda s: Logistic(), "linear", "logistic SGD"),
}
IDENTITY_PAIRS = (
    ("dense_sgd", "sparse_sgd"),
    ("local_hybrid", "sparse_local_hybrid"),
    ("local_hybrid", "sparse_shared_index"),
)


# ---------------------------------------------------------------- work model

def training_work(arm: str, k: int) -> tuple[int, int]:
    """(forward weight terms, W1+W2 weight writes) for one training step.

    Analytic source-level counts from the loop structure; biases excluded.
    """
    dense_fwd, sparse_fwd = INPUT_SIZE * HIDDEN_SIZE + 16, 8 * k + 16
    if arm == "dense_sgd":
        return dense_fwd, INPUT_SIZE * HIDDEN_SIZE + 16
    if arm == "sparse_sgd":
        return sparse_fwd, 8 * k + 16
    if arm in ("local_hybrid", "readout_only"):
        return dense_fwd, 8 * k + 16
    if arm in ("sparse_local_hybrid", "sparse_shared_index"):
        return sparse_fwd, 8 * k + 16
    if arm == "logistic":
        return k, k
    raise KeyError(arm)


# ---------------------------------------------------------------- protocol

def accuracy(model, examples) -> float:
    return sum(model.predict(e.features) == e.label for e in examples) / len(examples)


def weights(model) -> list[float]:
    if isinstance(model, Logistic):
        return model.w + [model.b]
    return ([v for row in model.w1 for v in row] + list(model.b1)
            + [v for row in model.w2 for v in row] + list(model.b2))


def run_arm(arm: str, seed: int, splits, epochs: int) -> dict:
    model = ARMS[arm][0](seed)
    matrix = [[accuracy(model, s.test) for s in splits]]  # row 0 = before training
    fwd = writes = 0
    for split in splits:
        for _ in range(epochs):
            for e in split.train:
                model.step(e.features, e.label)
                f, w = training_work(arm, sum(e.features))
                fwd += f
                writes += w
        matrix.append([accuracy(model, s.test) for s in splits])
    final = matrix[-1]
    # Rows: 0 = before training, r = after task r. Forgetting of task t = best
    # accuracy on it after it was learned minus final accuracy (>= 0).
    forgetting = [max(matrix[r][t] for r in range(t + 1, 5)) - final[t] for t in range(3)]
    return {
        "accuracy_matrix": matrix,
        "final_per_task": final,
        "final_mean": sum(final) / 4,
        "mean_forgetting_T1_T3": sum(forgetting) / 3,
        "train_forward_weight_terms": fwd,
        "train_weight_writes": writes,
        "_weights": weights(model),
    }


def local_rule_diagnostics(seed: int, splits, epochs: int) -> list[dict]:
    """Per task, for local_hybrid: bias size, hidden spread, predicted classes."""
    model = LocalHebbianMLP(seed=seed)
    test = [e.features for s in splits for e in s.test]
    out = []
    for split in splits:
        for _ in range(epochs):
            for e in split.train:
                model.step(e.features, e.label)
        hidden = [model._forward(model._input(x))[0] for x in test]
        out.append({
            "max_abs_b1": max(abs(b) for b in model.b1),
            "max_hidden_std_over_test_inputs": max(statistics.pstdev(h[j] for h in hidden)
                                                   for j in range(HIDDEN_SIZE)),
            "predicted_classes": sorted({model.predict(x) for x in test}),
        })
    return out


def or_rule_accuracy(examples) -> float:
    return sum((e.features[0] | e.features[1]) == e.label for e in examples) / len(examples)


def majority_accuracy(train, test) -> float:
    ones = sum(e.label for e in train)
    guess = int(ones * 2 > len(train))
    return sum(e.label == guess for e in test) / len(test)


def bootstrap_ci(values, rng) -> list[float]:
    n = len(values)
    means = sorted(sum(rng.choice(values) for _ in range(n)) / n
                   for _ in range(BOOTSTRAP_RESAMPLES))
    return [means[int(0.025 * BOOTSTRAP_RESAMPLES)], means[int(0.975 * BOOTSTRAP_RESAMPLES) - 1]]


def summarize(values, rng) -> dict:
    return {"mean": statistics.fmean(values), "ci95": bootstrap_ci(values, rng),
            "min": min(values), "max": max(values)}


def run(seeds=SEEDS, epoch_settings=EPOCH_SETTINGS) -> tuple[dict, dict]:
    rng = random.Random(0)
    results = {"experiment": "CL-50", "data_version": DATA_VERSION, "seeds": list(seeds),
               "python": sys.version.split()[0], "settings": {}, "dataset": {}}
    timing = {}
    data = {s: generate_shared_rule_candidate(s) for s in seeds}
    for s in seeds:
        splits = data[s]
        results["dataset"][str(s)] = {
            "sha256": shared_rule_dataset_sha256(splits),
            "mean_active_train": [statistics.fmean(sum(e.features) for e in t.train) for t in splits],
            "or_rule_test_accuracy": [or_rule_accuracy(t.test) for t in splits],
            "majority_test_accuracy": [majority_accuracy(t.train, t.test) for t in splits],
        }
    for epochs in epoch_settings:
        per_arm = {arm: [] for arm in ARMS}
        timing[str(epochs)] = {}
        for arm in ARMS:
            start = time.perf_counter()
            for s in seeds:
                per_arm[arm].append(run_arm(arm, s, data[s], epochs))
            timing[str(epochs)][arm] = time.perf_counter() - start
        identity = {
            f"{a}=={b}": all(x["_weights"] == y["_weights"] and x["accuracy_matrix"] == y["accuracy_matrix"]
                             for x, y in zip(per_arm[a], per_arm[b]))
            for a, b in IDENTITY_PAIRS
        }
        arms_out = {}
        for arm, runs in per_arm.items():
            arms_out[arm] = {
                "traversal": ARMS[arm][1], "rule": ARMS[arm][2],
                "final_mean": summarize([r["final_mean"] for r in runs], rng),
                "final_per_task_mean": [statistics.fmean(r["final_per_task"][t] for r in runs) for t in range(4)],
                "mean_forgetting_T1_T3": summarize([r["mean_forgetting_T1_T3"] for r in runs], rng),
                "seeds_with_final_mean_ge_0.65": sum(r["final_mean"] >= 0.65 for r in runs),
                "train_forward_weight_terms_mean": statistics.fmean(r["train_forward_weight_terms"] for r in runs),
                "train_weight_writes_mean": statistics.fmean(r["train_weight_writes"] for r in runs),
            }
        or_final = [statistics.fmean(results["dataset"][str(s)]["or_rule_test_accuracy"]) for s in seeds]
        paired = {}
        for a, b in (("dense_sgd", "logistic"), ("local_hybrid", "readout_only"),
                     ("dense_sgd", "local_hybrid"), ("dense_sgd", "readout_only")):
            diffs = [x["final_mean"] - y["final_mean"] for x, y in zip(per_arm[a], per_arm[b])]
            paired[f"{a} - {b}"] = summarize(diffs, rng)
        for a in ("dense_sgd", "local_hybrid", "logistic"):
            diffs = [x["final_mean"] - o for x, o in zip(per_arm[a], or_final)]
            paired[f"{a} - or_rule"] = summarize(diffs, rng)
        diags = [local_rule_diagnostics(s, data[s], epochs) for s in seeds]
        local_diag = [{
            "mean_max_abs_b1": statistics.fmean(d[t]["max_abs_b1"] for d in diags),
            "mean_max_hidden_std": statistics.fmean(d[t]["max_hidden_std_over_test_inputs"] for d in diags),
            "seeds_predicting_one_class": sum(len(d[t]["predicted_classes"]) == 1 for d in diags),
        } for t in range(4)]
        results["settings"][str(epochs)] = {
            "epochs_per_task": epochs, "arms": arms_out,
            "bitwise_identity": identity, "paired_final_mean_differences": paired,
            "local_hybrid_diagnostics_after_each_task": local_diag,
        }
    results["baselines"] = {
        "chance": 0.5,
        "majority": summarize([statistics.fmean(results["dataset"][str(s)]["majority_test_accuracy"])
                               for s in seeds], rng),
        "or_rule": summarize([statistics.fmean(results["dataset"][str(s)]["or_rule_test_accuracy"])
                              for s in seeds], rng),
        "or_rule_per_task_mean": [statistics.fmean(results["dataset"][str(s)]["or_rule_test_accuracy"][t]
                                                   for s in seeds) for t in range(4)],
        "xor_oracle": 1.0,
        "mean_active_train_per_task": [statistics.fmean(results["dataset"][str(s)]["mean_active_train"][t]
                                                        for s in seeds) for t in range(4)],
    }
    return results, timing


def main() -> None:
    results, timing = run()
    (HERE / "results.json").write_text(json.dumps(results, indent=1, sort_keys=True) + "\n")
    (HERE / "timing.json").write_text(json.dumps(timing, indent=1, sort_keys=True) + "\n")
    for epochs, block in results["settings"].items():
        print(f"\n== epochs per task: {epochs} ==  identity: {block['bitwise_identity']}")
        for arm, r in block["arms"].items():
            m = r["final_mean"]
            print(f"{arm:20s} final {m['mean']:.3f} [{m['ci95'][0]:.3f},{m['ci95'][1]:.3f}] "
                  f"per-task {[round(v, 3) for v in r['final_per_task_mean']]} "
                  f"forget {r['mean_forgetting_T1_T3']['mean']:.3f} "
                  f">=0.65: {r['seeds_with_final_mean_ge_0.65']}/{len(results['seeds'])} "
                  f"fwd {r['train_forward_weight_terms_mean']:.0f} writes {r['train_weight_writes_mean']:.0f}")
        for name, d in block["paired_final_mean_differences"].items():
            print(f"  {name:28s} {d['mean']:+.3f} [{d['ci95'][0]:+.3f},{d['ci95'][1]:+.3f}]")
        for t, d in enumerate(block["local_hybrid_diagnostics_after_each_task"]):
            print(f"  local_hybrid after T{t + 1}: |b1| {d['mean_max_abs_b1']:.2f} "
                  f"hidden std {d['mean_max_hidden_std']:.5f} "
                  f"one-class seeds {d['seeds_predicting_one_class']}/{len(results['seeds'])}")
    b = results["baselines"]
    print(f"\nOR rule {b['or_rule']['mean']:.3f} per task {[round(v, 3) for v in b['or_rule_per_task_mean']]}; "
          f"majority {b['majority']['mean']:.3f}; active/task {[round(v, 2) for v in b['mean_active_train_per_task']]}")


if __name__ == "__main__":
    main()
