# CL-64 — Screening Q.ANT designs with the accumulation simulator

**Tier 0** under [docs/RISK_TIERS.md](../../docs/RISK_TIERS.md). It runs locally in about 10 minutes on 4 processes and makes no Q.ANT run. It reuses the CL-58 Fashion-MNIST download and compares against the archived Exp027 Q.ANT results. **Exploratory.** [Predictions](PREDICTIONS.md) were pushed before the run (commit `23f12e3`). Physical energy: `NOT_MEASURED`.

Reproduce from the repository root: `python -m experiments.cl64_simulator_design_screen.run`, which writes [`results.json`](results.json).

## Question

The simulator (sequential bfloat16 accumulation, [CL-62](../cl62_qant_accumulation_simulator/README.md)/[CL-63](../cl63_simulator_out_of_sample/README.md)) predicted archived Q.ANT CPU-backend results. **Used as a local design tool, which Fourier/KAN design gives the best trade-off between accuracy and Q.ANT mismatch? Does the simulator also reproduce the archived Exp027 hierarchical results?**

## Method

- **Grouped monolithic.**
  - The CL-58 grid-4 models (784 → 10, 3 seeds) are evaluated with the logit sum split into g contiguous Q.ANT calls, g = 1–784.
  - Group outputs are summed in float32, and a bfloat16 bias is added at the end.
- **Hierarchical (Exp027 designs).**
  - g ∈ {8, 16, 28} blocks map 784/g pixels to w ∈ {2, 8} features, and a head maps the g·w features to 10.
  - The designs are reimplemented and trained in NumPy, following Exp027's initialisation and training recipe (3 seeds). Parameter counts match Exp027 exactly.
  - Every block and the head is one simulated Q.ANT call.
  - **Diagnostic:** one stage is simulated while the other is exact.
- **Simulator,** as in CL-62: each Q.ANT call has bfloat16 operands, cosines and products, a bfloat16 running sum and a bfloat16 bias. Disagreements are counted against the float32 forward pass, out of 10,000 test images.

## Results

**Grouped monolithic** (reference accuracy 0.8330 at every g)

| g (calls) | 1 | 2 | 4 | 8 | 16 | 49 | 98 | 196 | 784 |
|---|---|---|---|---|---|---|---|---|---|
| Simulated disagreements | 337.0 | 208.3 | 116.3 | 67.3 | 42.7 | 24.7 | 20.0 | 17.7 | 17.0 |
| Simulated Q.ANT-path accuracy | 0.8284 | 0.8314 | 0.8332 | 0.8333 | 0.8339 | 0.8334 | 0.8331 | 0.8333 | 0.8332 |

The float32-accumulation floor is 15–20 at every g. The log–log slope over g = 1–49 is −0.69.

**Hierarchical** (means of 3 seeds; the Exp027 columns are archived Q.ANT CPU-backend results)

| Design | Calls | Reference accuracy, NumPy / Exp027 | Simulated disagreements | Exp027 Q.ANT disagreements | Ratio | Stage 1 only / stage 2 only |
|---|---|---|---|---|---|---|
| 8×2 | 9 | 0.8178 / 0.8140 | 128.3 | 121.0 | 1.06 | 115.0 / 42.3 |
| 8×8 | 9 | 0.8452 / 0.8481 | 116.0 | 124.0 | 0.94 | 87.3 / 75.7 |
| 16×2 | 17 | 0.8315 / 0.8312 | 93.3 | 110.0 | 0.85 | 76.3 / 56.7 |
| 16×8 | 17 | 0.8542 / 0.8546 | 120.0 | 118.0 | 1.02 | 62.7 / 103.0 |
| 28×2 | 29 | 0.8394 / 0.8417 | 111.0 | 118.3 | 0.94 | 60.0 / 84.3 |
| **28×8** | 29 | **0.8598 / 0.8607** | 147.7 | 138.0 | 1.07 | 35.3 / 135.0 |

The simulated Q.ANT-path accuracy for 28×8 is 0.8580; the archived Exp027 Q.ANT accuracy is 0.8607.

## Predictions

| # | Prediction | Outcome |
|---|---|---|
| 1 | NumPy reference accuracies within ±1.5 pp of Exp027 | ✓ within 0.4 pp for all 6 |
| 2 | **Simulated disagreements within ±30% of Exp027 for ≥ 5 of 6 designs** | ✓ 6 of 6 (0.85–1.07×) |
| 3 | Simulated disagreements vary across designs by ≤ 1.5× | ✗ 1.58× (93 to 148; archive 1.25×) |
| 4 | Grouped: monotonic, with slope between −1.0 and −0.5 | ✓ −0.69 |
| 5 | At g = 784, `seq` within 2× of float32 accumulation | ✓ 17.0 against 17.0 |
| 6 | Grouped g ≥ 16 has fewer disagreements than every hierarchical design; 28×8 has the best accuracy | ✓ 43 or fewer against 93 or more; 0.858 against at most 0.834 |

**5 of 6 held.**

## Findings

1. **The simulator predicts a third archived experiment.** Across all six Exp027 hierarchical designs, simulated disagreements are within 0.85–1.07× of Q.ANT. The models are independent NumPy reimplementations, and their reference accuracies are within 0.4 pp of Exp027's.
2. **Splitting the sum into more Q.ANT calls is the cheapest fix, but only for mismatch.**
   - On a fixed monolithic model, 16 calls cut disagreements 8× (337 to 43), and 49 calls reach the rounding floor (about 17–25).
   - Accuracy does not change, because the model is unchanged.
3. **The hierarchical designs are more accurate (up to 0.86 against 0.83) but keep more mismatch (93–148).** The stage diagnostic shows why:
   - With narrow blocks (w=2), stage 1 dominates.
   - With wide blocks (w=8), the head dominates, because it sums g·w·4 terms sequentially (896 for 28×8). That explains the archive's puzzle that mismatch is nearly flat in stage-1 fan-in: making stage 1 smaller makes the head larger.
4. **Concrete design implication (a hypothesis, not tested):** a 28×8 hierarchical model whose *head* is also split into several Q.ANT calls should keep its 0.86 accuracy and move towards the grouped-monolithic mismatch floor. That design and one confirmatory Q.ANT run (Tier 2) are the natural next step.

## What this does and does not show

The Exp027 comparison is an aggregate match on reimplemented models, not a per-image check. The screened designs are only those of Exp022 and Exp027; the grouped-head design was not run. Everything concerns the CPU backend, not photonic hardware.
