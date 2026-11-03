# CL-52 — Cheap selection of active hidden units

**Tier 0** under [docs/RISK_TIERS.md](../../docs/RISK_TIERS.md): pure Python, offline, synthetic data, about 90 s of CPU in total. **Exploratory.** [Predictions](PREDICTIONS.md) were committed before the run (commit `d7a52dc`). Physical energy: `NOT_MEASURED`.

Reproduce from the repository root: `python -m experiments.cl52_cheap_winner_selection.run`. This writes [`results.json`](results.json), which is deterministic, and [`timing.json`](timing.json), which records wall time from a single run.

## Question

CL-51 found that a hidden layer with H=32 units, of which the top k=4 are kept, reaches 0.992 accuracy. But choosing those 4 units means computing all 32 pre-activations, which costs 3× the forward work of a dense H=8 network. **Can the winners be chosen more cheaply without losing accuracy?**

## Setup

- **Protocol:** shared-rule dataset, 30 paired seeds, 10 passes per task, backprop.
- **Correctness check:** the `exact` and `dense_H8` arms reproduce the CL-51 model bit for bit (checked in the run).

**Forward work per training step** counts:
- screening and hashing operations, including the cost of refreshing a selector every 128 steps, spread over those steps;
- the W1 terms computed exactly;
- the W2 terms.

**Not counted:** the one-off set-up cost of a selector, and bucket lookups, which a hash table makes O(1).

**Recall** is the share of the true top-4 units that the selector found, measured on test inputs after training. For `static_first_k` it is not meaningful, because only its 4 units are ever trained.

## Results

The table shows accuracy as the mean over 30 seeds with a bootstrap 95% CI. Work is relative to `exact`, which costs 179.5 terms per training step.

| Arm | Accuracy | Top-4 recall | Forward work | Prediction |
|---|---|---|---|---|
| `exact` (all 32 computed) | 0.992 [0.983, 0.998] | 1.00 | 1.00 | ≈ 0.99 ✓ |
| `dense_H8` | 0.999 [0.999, 1.000] | — | 0.33 | ≈ 0.999, ≈ 0.33 ✓ |
| `static_first_k` (= dense H=4) | **0.995** [0.987, 1.000] | — | **0.16** | ≥ 0.95, ≈ 0.15 ✓ |
| `random` | 0.693 [0.657, 0.730] | 0.13 | 0.16 | 0.60–0.80 ✓ |
| `lowrank_r2_c8` | 0.978 [0.960, 0.993] | 0.93 | 0.92 | ≥ 0.97 ✓, ≤ 0.60 ✗ |
| `lowrank_r4_c8` | 0.987 [0.975, 0.996] | 0.96 | 1.57 | — |
| `simhash_K3_L2` | 0.760 [0.729, 0.788] | 0.41 | 0.60 | 0.85–0.97 ✗ |
| `simhash_K2_L4` | 0.964 [0.952, 0.974] | 0.85 | 1.16 | 0.85–0.97 ✓, 0.4–0.8 ✗ |

## Findings

1. **The main prediction held: no cheap selector beats a small dense network.** A dense network with 4 hidden units matches `exact` (+0.004 [−0.007, +0.014]) with 16% of the work. Every selector that comes close to it in accuracy costs 5.6–9.6× more work.
2. **Screening costs about as much as it saves.** With sparse binary inputs (about 5.4 active), computing one unit exactly costs only about 5 terms. A selector that scores every unit (low-rank: H×r = 64 terms) or rehashes every unit weight vector costs as much as the exact computation it replaces. The low-rank selector's 115 screening terms per step break down into about 75 for scoring and about 41 for refreshing.
3. **Which unit is chosen matters less than staying consistent.** Choosing 4 random units per input gives 0.693, while always using the same 4 units gives 0.995. On this task, input-dependent sparsity gives no advantage over a smaller network.
4. **Hash recall drives accuracy.** SimHash with few tables (recall 0.41) loses 0.23 in accuracy; with more tables (recall 0.85) it is close but more expensive than `exact`.

**Timing** (the archived run in `timing.json`, single run, Python): from 2.8 s (`static_first_k`) to 12.3 s (`lowrank_r4_c8`) per arm for 30 seeds. This follows the same ordering as the counted work, but it is too noisy for close comparisons.

## Implications for H1

- At this scale, sparse expansion **plus** selection is worse than simply using a smaller dense network. That is a clear negative result for H1 in the hidden-layer form tested here.
- Selection can only pay off when exact per-unit cost is much higher than screening cost: for example with dense, high-dimensional inputs (large fan-in), or with very large H where hash lookup is sublinear. Testing that requires a larger task and vectorised kernels (Tier 0/1). It is the natural next step.
