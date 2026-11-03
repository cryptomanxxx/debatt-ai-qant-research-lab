# CL-62 — Does low-precision accumulation explain the Q.ANT mismatch?

**Tier 0** under [docs/RISK_TIERS.md](../../docs/RISK_TIERS.md). It runs locally, reusing the CL-58 Fashion-MNIST download (pinned SHA-256), and compares against the archived Q.ANT counts from CL-58. No Q.ANT run was made. 4 processes, 19 minutes. **Exploratory.** [Predictions](PREDICTIONS.md) were pushed before the run (commit `8e52555`). Physical energy: `NOT_MEASURED`.

Reproduce from the repository root: `python -m experiments.cl62_qant_accumulation_simulator.run`, which writes [`results.json`](results.json).

## Question

In [CL-58](../cl58_qant_quantization_aware_training/README.md), the Q.ANT CPU backend disagreed with the float32 reference on 260–535 of 10k test images, and bfloat16 operand rounding explained only 2–8 of them. **Does a simulator with low-precision accumulation reproduce the Q.ANT disagreement counts?**

## Method

1. **Retrain the 18 CL-58 models** (grids 4/8 × 3 arms × 3 seeds) with unchanged CL-58 code. Reference accuracies and CL-58 simulator counts were reproduced **exactly** in all 18 runs.
2. **Evaluate five simulator variants,** fixed in advance. All of them start from CL-58's bfloat16 operand rounding and also round the output and bias to bfloat16.
   - `out`: float32 accumulation.
   - `seq`: the running sum is rounded to bfloat16 after every term, with terms in pixel-major order.
   - `pair`: a pairwise tree in bfloat16.
   - `block32` and `block256`: float32 blocks with bfloat16 block sums.
3. **Compare with Q.ANT.** Disagreements with the reference are compared against the archived Q.ANT counts, run by run.

## Results

Means over 9 runs per grid; disagreements are with the float32 reference, out of 10,000 test images.

| Variant | Grid 4 disagreements | Grid 8 disagreements | Accuracy (g4 / g8) | r with Q.ANT, within grid (g4 / g8) |
|---|---|---|---|---|
| **Q.ANT CPU backend (archived)** | **330.4** | **465.2** | **0.8291 / 0.8335** | — |
| CL-58 simulator | 5.8 | 5.4 | 0.8331 / 0.8427 | −0.15 / +0.31 |
| `out` | 16.7 | 14.8 | 0.8332 / 0.8429 | +0.70 / +0.18 |
| **`seq`** | **332.9** | **465.1** | **0.8291 / 0.8334** | **+0.99 / +0.98** |
| `pair` | 24.1 | 20.4 | 0.8331 / 0.8427 | +0.18 / −0.01 |
| `block32` | 16.2 | 14.1 | 0.8332 / 0.8428 | +0.48 / +0.45 |
| `block256` | 16.8 | 14.3 | 0.8331 / 0.8428 | +0.50 / +0.54 |

Run by run, `seq` differs from Q.ANT by 0–19 disagreements (mean absolute difference 1.7% for grid 4 and 2.1% for grid 8). Its accuracy differs by at most 0.0013. Over all 18 runs, its correlation with the Q.ANT counts is 0.99.

## Predictions

| # | Prediction | Outcome |
|---|---|---|
| 1 | Retrained models reproduce CL-58 exactly | ✓ all 18 runs |
| 2 | Output rounding alone gives < 30% of the Q.ANT counts | ✓ 3–5% |
| 3 | `seq` gives > 3× the Q.ANT counts | ✗ **it matches them (1.00–1.01×)** |
| 4 | **`pair` or a block variant within ±20% for both grids** | ✗ 3–7% |
| 5 | That variant correlates ≥ 0.8 with Q.ANT across runs | ✗ no variant met #4 (`seq` reaches 0.99) |

**2 of 5 held.** My paper reasoning for prediction 3 used the worst-case error bound, n·u. In this layer the running sum stays small, because terms have both signs, so the typical error is much smaller.

## Findings

1. **Sequential bfloat16 accumulation reproduces the Q.ANT mismatch in aggregate.** It matches the mean disagreement counts to within 1% for both grids, matches the accuracies to within 0.0001 on average, and tracks the variation between runs (r = 0.98–0.99 within each grid).
   - None of the other schemes comes close: float32, pairwise and blockwise give only 3–7% of the counts.
   - The 18 runs differ in model, not data. That the variation between runs tracks this closely is the strongest evidence here.
2. **The remaining run-by-run differences (up to 19 of 10k) mean the simulator is not bit-exact.** The order of summation or another detail probably differs; this was not explored after the run, to avoid fitting.
3. **This turns the CL-58 hypothesis into a concrete, testable model:** the Q.ANT CPU backend behaves as if each logit is accumulated term by term in bfloat16. That would explain the earlier architectural finding (Exp021–030) that lower fan-in and hierarchical accumulation reduce the mismatch, and why QAT on operands could not help.

## What this does and does not show

It shows a close aggregate match on 18 models and two grid sizes. It does **not** show that the same images flip: CL-58 archived counts, not per-image predictions. A per-image check needs a new Q.ANT run that saves predictions (Tier 2, workflow dispatch). The finding concerns the CPU backend, not photonic hardware.
