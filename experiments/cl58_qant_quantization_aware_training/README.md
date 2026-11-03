# CL-58 — Quantization-aware training for the Q.ANT Fourier/KAN layer

- **Part 1:** local, Tier 1. It downloads Fashion-MNIST; the SHA-256 is pinned and the MD5 matches torchvision's official sums.
- **Part 2:** Tier 2, owner-approved. The manual workflow [`cl58-qant-qat.yml`](../../.github/workflows/cl58-qant-qat.yml) ran the same code in the Q.ANT research image on the **Q.ANT CPU backend** ([run 41656287068](https://github.com/cryptomanxxx/debatt-ai-qant-research-lab/actions/runs/41656287068), 14 minutes).

**Exploratory.** [Predictions](PREDICTIONS.md) were pushed before any run (commit `f4dae0a`). A 1,280-sample development run was seen before the full runs; nothing was changed after it. Physical energy: `NOT_MEASURED`. These are CPU-backend results, not photonic hardware results.

Reproduce from the repository root:
- Part 1: `python -m experiments.cl58_qant_quantization_aware_training.run`, which writes [`results_local.json`](results_local.json).
- Part 2: dispatch the workflow; it adds `--qant`. [`results_qant.json`](results_qant.json) is the run's artifact `cl58-results-qant`, unchanged. The owner supplied it; its zip SHA-256 matches the digest the run logged (`7acc1033…`). [`results_qant_summary.json`](results_qant_summary.json) records the run, job and artifact identifiers.

## Question

Models trained with an exact cosine disagree with their float reference on 300–600 of 10k test images when evaluated on the Q.ANT CPU backend (Exp015). Training against a fitted Q.ANT response curve did not help (Exp015). **Does quantization-aware training (QAT), which rounds to bfloat16 in the forward pass during training, reduce that mismatch?**

## Setup

- **Model:** the Exp015 Fourier/KAN layer, `logits = b + Σ A·cos(k·x + φ)`, 784 → 10, grid 4 and 8, with Exp015's initialisation.
- **Training:** Adam (lr 1e-3, batch 128), 3 epochs, seeds 11, 22 and 33.
- **Arms:**
  - `exact_cosine`
  - `bf16_ste`: inputs, phases, amplitudes, bias and cosine outputs are rounded to bfloat16 in the forward pass, with straight-through gradients.
  - `bf16_ste_noise`: as `bf16_ste`, plus σ=0.003 noise on the cosine outputs during training.
- **Evaluation:**
  - float32 reference accuracy;
  - a bfloat16 simulator that rounds the same quantities and sums in float32 (it matches `ml_dtypes` exactly, as tested in the image);
  - Q.ANT CPU-backend accuracy via `calc_kan_layer_fprop` + `add_bias_fprop`, as in Exp015.

The reference and simulator numbers are identical between the local run and the Actions run, which used different machines and NumPy 2.5.3 against 2.3.5.

## Results

Mean over 3 seeds; disagreements are out of 10,000 test images.

| Grid, arm | Reference accuracy | Q.ANT accuracy | Disagreements: reference vs simulator | Disagreements: reference vs Q.ANT | Disagreements: simulator vs Q.ANT |
|---|---|---|---|---|---|
| 4, `exact_cosine` | 0.8330 | 0.8285 | 6.3 | 332.0 | 334.3 |
| 4, `bf16_ste` | 0.8331 | 0.8294 | 4.3 | 334.7 | 334.3 |
| 4, `bf16_ste_noise` | 0.8330 | 0.8293 | 6.7 | 324.7 | 328.7 |
| 8, `exact_cosine` | 0.8429 | 0.8334 | 5.0 | 462.0 | 462.3 |
| 8, `bf16_ste` | 0.8429 | 0.8330 | 5.7 | 465.3 | 463.3 |
| 8, `bf16_ste_noise` | 0.8429 | 0.8341 | 5.7 | 468.3 | 468.0 |

## Predictions

| # | Prediction | Outcome |
|---|---|---|
| 1 | `exact_cosine` Q.ANT accuracy within 0.01 of Exp015 (0.830 / 0.821) | ✗ grid 4 ✓ (0.8285); grid 8 ✗ (0.8334, +0.012) |
| 2 | `exact_cosine` disagreements within ±30% of Exp015 (323 / 587) | ✓ 332 / 462 |
| 3 | Simulated bfloat16 explains 30–70% of the Q.ANT disagreements | ✗ about 1–2% (5–7 against 324–468) |
| 4 | QAT halves simulated disagreements | ✗ no effect (about 5 in every arm) |
| 5 | **QAT cuts Q.ANT disagreements by ≥ 30% for grid 8** | ✗ no effect (465 against 462) |
| 6 | Noise cuts Q.ANT disagreements more than QAT alone (grid 8) | ✗ 468 against 465 |
| 7 | QAT keeps reference accuracy within 0.01 | ✓ differences ≤ 0.0001 |

**2 of 7 held.**

## Findings

1. **QAT does not reduce the Q.ANT mismatch.** Disagreements with the float reference are unchanged for both grids (mean differences ≤ 10 of 10,000; grid 4 324.7–334.7, grid 8 462.0–468.3), and Q.ANT accuracy moves by at most 0.0009. Adding training noise does not help either.
   Seed by seed, the change in disagreements against `exact_cosine` has no consistent sign:
   - Grid 4, QAT: +12, +10, −14. Grid 4, QAT + noise: −15, +12, −19.
   - Grid 8, QAT: +9, +2, −1. Grid 8, QAT + noise: +12, +6, +1.
2. **The Q.ANT mismatch is not caused by bfloat16 rounding of the operands.** Rounding inputs, phases, amplitudes, bias and cosine outputs explains about 1–2% of the disagreements. The simulator disagrees with Q.ANT almost exactly as often as the float reference does.
3. **Training-side fixes have now failed twice:** Exp015 (fitted response curve) and CL-58 (bfloat16 QAT ± noise). The lever the lab found earlier was architectural: lower fan-in and hierarchical accumulation (Exp021–030).

## What this suggests (hypotheses, not tested here)

The remaining mismatch is consistent with lower-precision **accumulation** of the 3,136 (grid 4) or 6,272 (grid 8) terms per logit, in line with the Exp021 accumulation scaling law. It could also be the Q.ANT-specific residual seen in Exp014. A direct next test would add accumulation in bfloat16 (or blockwise) to the simulator and check whether it reproduces the Q.ANT disagreements. If it does, QAT against that simulator, or the existing low-fan-in architectures, can be compared fairly.
