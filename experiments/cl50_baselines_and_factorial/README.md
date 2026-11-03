# CL-50 — Baselines and dense/sparse × backprop/local comparison

**Tier 0** under [docs/RISK_TIERS.md](../../docs/RISK_TIERS.md): pure Python, offline, synthetic data, about 50 s of CPU in total. **Exploratory result, not preregistered.** Physical energy: `NOT_MEASURED`.

Reproduce from the repository root: `python -m experiments.cl50_baselines_and_factorial.run`. This writes [`results.json`](results.json), which is deterministic, and [`timing.json`](timing.json), which records wall time from a single run.

## Question

1. Do the CL arms learn the shared-rule task at all, compared with simple baselines?
2. Does sparse traversal change anything apart from the work count?

## Setup

- **Data:** the shared-rule candidate (`cl2-shared-rule-candidate-v0`), with label `x0 XOR x1`. There are four tasks with input activity rising from 0.125 to 0.5. Each task has 128 training and 64 test examples, and tasks are learned sequentially with no replay. Per-seed SHA-256 digests are in `results.json`.
- **Seeds:** 30 paired seeds (101–130). Every MLP arm starts from identical initial weights.
- **Training:** 1 pass per task (the CL-3X draft setting) and 10 passes per task (exploratory). Default learning rates are 0.05 for SGD and 0.01 for the local rule.
- **Arms:** `dense_sgd`, `sparse_sgd` (new: backprop that only writes to active W1 columns), `local_hybrid`, `sparse_local_hybrid`, `sparse_shared_index`, `readout_only` (new control: hidden layer frozen at its random initial values), and `logistic` (new: a linear model on the raw inputs).
- **Baselines:** chance, majority class, the rule `x0 OR x1`, and an XOR oracle.

## Results

Final mean test accuracy over the four tasks, as the mean over 30 seeds with a bootstrap 95% CI:

| Arm | 1 pass / task | 10 passes / task | Seeds ≥ 0.65 (1 / 10) |
|---|---|---|---|
| `dense_sgd` = `sparse_sgd` | 0.636 [0.58, 0.69] | **0.999** [0.999, 1.000] | 12 / 30 |
| `local_hybrid` = `sparse_local_hybrid` = `sparse_shared_index` | **0.500** [0.500, 0.500] | **0.500** [0.500, 0.500] | 0 / 0 |
| `readout_only` (frozen random hidden layer) | 0.542 [0.52, 0.56] | 0.541 [0.52, 0.56] | 2 / 3 |
| `logistic` (linear model) | 0.626 [0.58, 0.67] | 0.516 [0.50, 0.53] | 13 / 0 |
| `x0 OR x1` rule (no training) | 0.882 | 0.882 | — |
| chance / majority | 0.500 / 0.500 | | — |

Per task, the OR rule scores 0.99, 0.94, 0.86 and 0.75. Mean active inputs per task are 2.8, 4.5, 6.2 and 8.0.

### Findings

1. **Sparse traversal is bit-for-bit identical.** `dense_sgd` and `sparse_sgd` end with exactly the same weights and the same accuracy matrices in all 60 runs, and so do the three local arms. Sparse traversal therefore never changes quality on binary inputs (as predicted in CL-49), and this holds for backprop too. Its only effect is on work: about 59% fewer forward weight terms in training (30,138 against 73,728 per seed and pass) and, for SGD, 59% fewer weight writes.
2. **The local hidden-layer rule fails completely.** In every seed and both settings, the local arms predict a single class for all 256 test inputs after every task. The rule `w += η·h`, `b += η·h` makes the biases grow (max |b1| reaches 4.2 after one pass and 50 after ten passes). The hidden units saturate at a constant ±1, and spread across inputs falls from 0.26 after T1 to 0.0003 after T4 (1 pass). The local rule even does worse than a frozen random hidden layer (−0.042 [−0.061, −0.024]).
3. **Backprop learns XOR when trained enough.** With 10 passes, `dense_sgd` reaches 0.999 on every task, beating the OR rule by +0.118 [+0.111, +0.124]. One pass is clearly too little training: 0.636 is below the OR rule.
4. **There is almost no forgetting to study.** Under backprop, mean forgetting of T1–T3 is 0.001 with 10 passes. The shared rule involves only covariate shift, so this task cannot test H2.
5. **The 0.65 quality floor does not discriminate.** At 1 pass, both the undertrained MLP and a linear model cross it in some seeds, while the rule-based OR baseline (0.88) is far above it.

**Timing (one run, Python, no A/A control):** with 10 passes, `sparse_sgd` took 6.1 s and `dense_sgd` 12.8 s. `sparse_local_hybrid` took 6.4 s and `local_hybrid` 5.9 s. Fewer counted operations did not reliably mean less wall time. These numbers are indicative only.

## Implications

- The primary H1 comparison, `local_hybrid` against `sparse_local_hybrid`, cannot pass any quality gate as long as the local rule collapses. Under CL-9B rules it would be **INCONCLUSIVE**, whatever the work counts show.
- The local rule needs normalisation or competition (Oja, BCM, kWTA) or a label-modulated three-factor term before it can be compared with anything. That is the most valuable next step.
- A meaningful sparsity question needs computation that sparsity changes, such as hidden-layer kWTA or thresholding. Input sparsity on binary inputs only changes the work count.
- H2 needs a task with real interference.
