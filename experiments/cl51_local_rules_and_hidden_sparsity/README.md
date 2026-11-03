# CL-51 — Stabilised local rules and hidden-layer kWTA sparsity

**Tier 0** under [docs/RISK_TIERS.md](../../docs/RISK_TIERS.md): pure Python, offline, synthetic data, about 90 s of CPU in total. **Exploratory.** [Predictions](PREDICTIONS.md) were committed before the run (commit `39b35cc`). Physical energy: `NOT_MEASURED`.

Reproduce from the repository root: `python -m experiments.cl51_local_rules_and_hidden_sparsity.run`. This writes [`results.json`](results.json), which is deterministic, and [`timing.json`](timing.json), which records wall time from a single run.

## Questions

- **A.** Can any local hidden-layer rule learn the task, given that the original Hebbian rule collapses (CL-50)?
- **B.** Does sparsity in the hidden layer (kWTA) give a real trade-off between accuracy and work, unlike input sparsity, which changes nothing but the work count?

## Setup

- **Data and protocol:** same as CL-50. Shared-rule dataset, 30 paired seeds (101–130), **10 passes per task**, tasks learned sequentially. Physical energy is `NOT_MEASURED`.
- **Model:** one parameterised model, `GenMLP` (16 → H → 2, tanh). It reproduces `TinyDenseMLP` (backprop) and `LocalHebbianMLP` (Hebbian) **bit for bit** in all 30 seeds; the run checks this.
- **Readout:** every local rule uses the `local_hybrid` readout (delta rule).
- **kWTA:** keeps the k hidden units with the largest pre-activation, and the rest output 0. All pre-activations must still be computed in order to choose the winners.

## Results

Final mean test accuracy over T1–T4, as the mean over 30 seeds with a bootstrap 95% CI. The OR rule scores 0.882.

### A — local hidden-layer rules (H=8)

| Rule | Accuracy | Seeds predicting one class | Mean max \|w1\| | Prediction |
|---|---|---|---|---|
| backprop (reference) | 0.999 [0.998, 1.000] | 0 | 3.0 | — |
| `hebb` (original) | 0.500 [0.500, 0.500] | 30 | 19.5 | 0.500 ✓ |
| `hebb_nobias` | 0.500 [0.500, 0.500] | 30 | 19.1 | < 0.65 ✓ |
| `oja` (on active inputs) | 0.500 [0.500, 0.501] | 28 | 1.0 | 0.50–0.65 ✓ (low end) |
| `competitive_k2` (kWTA + instar) | 0.523 [0.513, 0.535] | 6 | 0.9 | 0.55–0.75 ✗ |
| `dfa` (direct feedback alignment) | **0.815** [0.769, 0.856] | 0 | 1.5 | > 0.90 ✗ |

### B — hidden-layer kWTA with backprop

Work columns are relative to dense backprop with H=8 (301,379 forward weight terms and as many weight writes per seed).

| H, k | Accuracy | Forward weight terms | W1+W2 weight writes | Prediction |
|---|---|---|---|---|
| 8, 8 (dense) | 0.999 [0.999, 1.000] | 1.00 | 1.00 | ≈ 0.999 ✓ |
| 8, 4 | 0.970 [0.933, 0.992] | 0.86 | 0.50 | ≥ 0.95 ✓ |
| 8, 2 | 0.912 [0.868, 0.948] | 0.80 | 0.25 | clearly lower ✓ |
| 8, 1 | 0.807 [0.753, 0.858] | 0.76 | 0.13 | ≤ OR rule ✓ |
| 32, 2 | 0.947 [0.909, 0.976] | 2.98 | 0.25 | — |
| 32, 4 | **0.992** [0.983, 0.998] | 3.05 | **0.50** | ≥ 0.95 ✓ |
| 32, 8 | 1.000 [1.000, 1.000] | 3.18 | 1.00 | — |
| 32, 32 (dense) | 1.000 [1.000, 1.000] | 4.00 | 4.00 | — |

## Findings

1. **None of the label-independent local rules learn the task.** Without bias updates the weights still grow without bound. Oja keeps the weights bounded (max |w| ≈ 1), but the model still predicts one class in 28 of 30 seeds. A likely reason, not checked here, is that the rule only sees active inputs and pushes their weights towards similar values. Competitive learning only just beats chance.
2. **Direct feedback alignment is the only local rule that works**: 0.815, which is +0.315 [+0.271, +0.357] better than Hebbian learning. It is still −0.185 [−0.231, −0.143] below backprop and below the OR rule. It sends a global error signal through fixed random weights, so it is "local" in a weaker sense.
3. **Hidden-layer kWTA gives a real accuracy–work trade-off.** At H=8, halving the active units (k=4) halves the weight writes and costs 0.030 [0.007, 0.066] accuracy. Unlike input sparsity, this changes the result, so it is a genuine sparsity question.
4. **Expansion plus kWTA has a cost that is easy to miss.** H=32 with k=4 nearly matches dense H=8 (−0.008 [−0.017, −0.001]) with half the weight writes. But because all 32 pre-activations are computed to choose the winners, it needs **3× more forward work**. Sparse expansion only pays off if winners can be chosen without computing every unit, for example through hashing as in SLIDE. That is a concrete next question.
5. **There is still almost no forgetting** (at most 0.12, with backprop near 0), as in CL-50.

**Timing** (the archived run in `timing.json`, single run, Python): between 2.2 s (competitive) and 10.1 s (H=32 dense) per arm for 30 seeds. Too noisy for comparisons between arms of similar cost.

## Implications for H1

- The only working local rule is DFA, so a meaningful H1 comparison with a "local rule" means `dfa` against `sparse dfa`. With binary inputs that comparison would again be bit-identical; only hidden-layer sparsity changes the result.
- The most promising H1 direction is **hidden-layer sparsity with cheap winner selection**. The counts here are analytic proxies; real savings need measurements on vectorised or compiled kernels.
