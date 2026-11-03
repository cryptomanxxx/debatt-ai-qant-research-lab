# CL-69 — A systematic model zoo for Q.ANT numerical fidelity

**Tier 0** under [docs/RISK_TIERS.md](../../docs/RISK_TIERS.md), reusing the CL-58 Fashion-MNIST download; on a fresh checkout that download comes first, which is Tier 1. It runs locally in about 11 minutes on 4 processes, simulator only; no Q.ANT run was made. This is phase C of the [research goal](../../docs/RESEARCH_ROADMAP.md#research-goal-set-by-the-owner-2026-11-03). **Exploratory.** [Predictions](PREDICTIONS.md) were pushed before the run (commit `6b5e5cc`). Physical energy: `NOT_MEASURED`.

Reproduce from the repository root: `python -m experiments.cl69_qant_model_zoo.run`, which writes [`results.json`](results.json).

## Question

**Which model types and properties keep their CPU/GPU (float32) accuracy best, and which lose it worst, on the Q.ANT CPU-backend path, and why?** Each property is varied one at a time.

## Method

- **Models.** 14 models on Fashion-MNIST: linear (logistic regression and ReLU/sigmoid MLPs of varied width and depth), KAN (grid 1–8, split calls, the CL-65 hierarchical design) and two hybrids. All use Adam lr 1e-3, batch 128, 3 epochs and seeds 11/22/33.
- **Q.ANT path, simulated with the evidence so far:**
  - `calc_kan_layer_fprop` uses a sequential bfloat16 sum, confirmed on Q.ANT per image in CL-66.
  - `linear_fprop` uses float32 accumulation, an aggregate match in CL-68.
  - All operands and outputs are bfloat16, and the final class is the argmax of the bfloat16 logits.
  - The sigmoid path (float32 with bfloat16 output) is an assumption, not checked on Q.ANT.
- **Reproduction check.** `kan_g4` reproduces CL-62's simulated disagreements exactly (337.0), because it has the same initialisation and data order. For reference, the archived Q.ANT counts for `kan_g1` and `kan_g2` were 159 and 274 (Exp010, with different torch models).

## Results

Means over 3 seeds. "Longest sum" is the longest sequential bfloat16 sum in any single call; 0 means float32 accumulation only.

| Model | Family | Parameters | Q.ANT calls | Longest sum | Float32 accuracy | Q.ANT-path accuracy | Gap (pp) | Disagreements per 10k |
|---|---|---|---|---|---|---|---|---|
| `logreg` | linear | 7,850 | 1 | 0 | 0.8331 | 0.8334 | +0.03 | 15.7 |
| `mlp_h64` | linear | 50,890 | 2 | 0 | 0.8484 | 0.8485 | +0.00 | 17.3 |
| `mlp_h256` (base) | linear | 203,530 | 2 | 0 | 0.8559 | 0.8558 | −0.01 | 8.7 |
| `mlp_h1024` | linear | 814,090 | 2 | 0 | 0.8658 | 0.8656 | −0.02 | 12.3 |
| `mlp_h256x2` | linear | 269,322 | 3 | 0 | 0.8618 | 0.8619 | +0.01 | **7.0** |
| `mlp_h256x3` | linear | 335,114 | 4 | 0 | **0.8670** | **0.8670** | −0.01 | 10.0 |
| `mlp_sig_h256` | linear | 203,530 | 2 | 0 | 0.8547 | 0.8552 | +0.04 | 11.3 |
| `kan_g1` | KAN | 15,690 | 1 | 784 | 0.8161 | 0.8159 | −0.02 | 162.3 |
| `kan_g2` | KAN | 31,370 | 1 | 1,568 | 0.8314 | 0.8280 | −0.34 | 258.0 |
| `kan_g4` (base) | KAN | 62,730 | 1 | 3,136 | 0.8330 | 0.8284 | −0.46 | 337.0 |
| **`kan_g8`** | KAN | 125,450 | 1 | 6,272 | 0.8429 | 0.8333 | **−0.96** | **470.7** |
| `kan_g4_split16` | KAN | 62,730 | 16 | 196 | 0.8330 | 0.8339 | +0.08 | 42.7 |
| `hier_28x8_head8` | KAN | 68,330 | 36 | 112 | 0.8598 | 0.8600 | +0.02 | 45.7 |
| `lin128_kanhead` | hybrid | 110,730 | 2 | 512 | 0.8499 | 0.8498 | −0.01 | 95.0 |
| `kan32_linhead` | hybrid | 201,066 | 2 | 3,136 | 0.8393 | 0.8393 | +0.00 | 129.7 |

Across the 8 KAN-containing models, the tie-corrected Spearman rank correlation between the longest sequential sum and the disagreements is 0.87. The first version reported 0.83 from a rank function that broke the tie between `kan_g4` and `kan32_linhead` (both 3,136 terms) by order; Codex caught it, and the value was recomputed from the stored summary without a rerun.

## Predictions

| # | Prediction | Outcome |
|---|---|---|
| 1 | Every linear model has < 20 disagreements and a gap within ±0.1 pp | ✓ 7.0–17.3 disagreements; gaps of −0.02 to +0.04 pp |
| 2 | Width and depth change linear disagreements by ≤ 2× from the base | ✓ 0.8–1.99× (h64 is at the limit) |
| 3 | KAN disagreements rise with grid: g1 < g2 < g4 < g8 | ✓ 162 < 258 < 337 < 471 |
| 4 | **Worst-suited model is `kan_g8`; the best-suited are linear** | ✓ `kan_g8` has the most disagreements and the largest gap; the fewest are in `mlp_h256x2` |
| 5 | `lin128_kanhead` ≥ 3× fewer disagreements than `kan_g4`; `kan32_linhead` keeps ≥ half of `kan_g4`'s | ✗ the first holds (3.5×), but `kan32_linhead` keeps only 38% |
| 6 | Spearman(longest sum, disagreements) ≥ 0.7 for the KAN-containing models | ✓ 0.87 (tie-corrected) |

**5 of 6 held.**

## Findings

1. **On the Q.ANT CPU backend, the primitive decides fidelity, not model size.**
   - All seven linear models lose essentially nothing: gaps within ±0.04 pp and 7–17 disagreements per 10k, at every width (64–1024), depth (1–3 hidden layers) and activation tested.
   - The KAN models range from about 40 to about 470 disagreements, and lose up to 0.96 pp of accuracy.
2. **Within KAN, the longest sequential bfloat16 sum per call explains the ranking** (ρ = 0.87). More grid components per input mean longer sums and more mismatch.
   - The biggest gap, and so the worst-suited model, is the largest monolithic KAN (`kan_g8`, 6,272-term sums).
   - Splitting calls (`kan_g4_split16`) or a hierarchical layout (`hier_28x8_head8`) brings KAN to about 45 disagreements with no accuracy loss.
3. **Hybrids sit in between, depending on where the KAN layer is.** A KAN head over 128 linear features gives 95 disagreements. A KAN first layer with a linear head gives 130, fewer than its 3,136-term sums predicted. A hypothesis, not tested: the linear head averages 32 noisy features, which partly cancels their errors.
4. **The best model on the Q.ANT path in this zoo is a plain 3-layer ReLU MLP** (0.867, with no loss compared with float32). The best KAN design reaches 0.860.

## Answer to the research question, so far

For **numerical fidelity on the Q.ANT CPU backend**:
- **Best suited:** models built on `linear_fprop` (MLPs), at any size tested.
- **Worst suited:** large monolithic KAN layers on `calc_kan_layer_fprop`.
- **Why:** the KAN primitive rounds its running sum to bfloat16 after every term, while the linear primitive accumulates in float32 (CL-62 to CL-68).

This says nothing about speed or energy on photonic hardware, which may favour different models. It also assumes that the CPU backend reflects the hardware's numerics, which is untested.

## What this does and does not show

All results are simulated on one dataset with 3 seeds. The `linear_fprop` model rests on an aggregate match (CL-68), and the sigmoid and softmax paths are assumptions. Phase B, a Q.ANT run that measures `linear_fprop` and the activations directly, would check these assumptions (Tier 2).
