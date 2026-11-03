# CL-58 — Quantization-aware training for the Q.ANT Fourier/KAN layer

**Part 1 (this PR):** Tier 1. It downloads Fashion-MNIST; the SHA-256 is pinned and the MD5 matches torchvision's official sums. It ran locally in NumPy, 3 processes, about 20 minutes.
**Part 2:** Tier 2, owner-approved. The manual workflow [`cl58-qant-qat.yml`](../../.github/workflows/cl58-qant-qat.yml) runs the same code in the Q.ANT image and evaluates on the Q.ANT CPU backend. It can only be dispatched once this workflow is on `main`, so **its results follow in a separate PR**.

**Exploratory.** [Predictions](PREDICTIONS.md) were committed and pushed before any run (commit `f4dae0a`). Physical energy: `NOT_MEASURED`. A development run (1,280 samples, one seed, all arms) was seen before the full run; nothing was changed after it.

Reproduce from the repository root:
- Part 1: `python -m experiments.cl58_qant_quantization_aware_training.run`, which writes [`results_local.json`](results_local.json).
- Part 2: run inside the Q.ANT image with `--qant`.

## Question

Models trained with an exact cosine disagree with their float reference on 300–600 of 10k test images when evaluated on the Q.ANT CPU backend (Exp015). Training against a fitted Q.ANT response curve did not help. **Does training that sees bfloat16 rounding during training (quantization-aware training, QAT) reduce the mismatch?**

## Setup

- **Model:** the Exp015 single-layer Fourier/KAN model, `logits = b + Σ A·cos(k·x + φ)`, 784 → 10, grid 4 and 8, with Exp015's initialisation.
- **Training:** Adam (lr 1e-3, batch 128), 3 epochs, seeds 11, 22 and 33.
- **Arms:**
  - `exact_cosine`
  - `bf16_ste`: inputs, phases, amplitudes, bias and cosine outputs are rounded to bfloat16 in the forward pass, with straight-through gradients.
  - `bf16_ste_noise`: as `bf16_ste`, plus σ=0.003 Gaussian noise on the cosine outputs during training.
- **Part 1 evaluation:** float32 reference accuracy; accuracy with the bfloat16 simulator, which rounds the same quantities but sums in float32; and disagreements between the two.

## Part 1 results (local, simulated bfloat16)

Mean over 3 seeds; disagreements are out of 10,000 test images.

| Grid, arm | Reference accuracy | Simulated-bf16 accuracy | Disagreements (reference vs simulated) |
|---|---|---|---|
| 4, `exact_cosine` | 0.8330 | 0.8331 | 6.3 |
| 4, `bf16_ste` | 0.8331 | 0.8332 | 4.3 |
| 4, `bf16_ste_noise` | 0.8330 | 0.8331 | 6.7 |
| 8, `exact_cosine` | 0.8429 | 0.8427 | 5.0 |
| 8, `bf16_ste` | 0.8429 | 0.8428 | 5.7 |
| 8, `bf16_ste_noise` | 0.8429 | 0.8428 | 5.7 |

For comparison, Exp015 measured 323 (grid 4) and 587 (grid 8) disagreements on the real Q.ANT CPU backend, at Q.ANT accuracy 0.830 and 0.821.

## Findings so far

1. **Rounding inputs, weights and cosine outputs to bfloat16 explains almost none of the Q.ANT mismatch:** about 5 of 10,000 disagreements, against 300–600 on real Q.ANT. Most of the Q.ANT error must come from elsewhere. Candidates are lower-precision accumulation of the ~3,000–6,000 summed terms per logit (consistent with the lab's Exp021 accumulation scaling law) or the Q.ANT-specific residual seen in Exp014.
2. **QAT has nothing to fix against this simulator.** Disagreements stay at 4–7 for every arm, so prediction 4 fails. This says nothing yet about real Q.ANT; Part 2 measures that directly.
3. **QAT costs no float accuracy:** reference accuracy is unchanged to the fourth decimal. Training is 1.5× slower with rounding and about 4× slower with noise, in NumPy.

## Predictions (status after Part 1)

| # | Prediction | Status |
|---|---|---|
| 1–3, 5, 6 | Real Q.ANT accuracy and disagreements | Pending (Part 2) |
| 4 | QAT cuts simulated-bf16 disagreements by ≥ 50% | ✗ No effect: about 5 in every arm, too few to reduce |
| 7 | Reference accuracy of QAT arms within 0.01 of `exact_cosine` | ✓ Differences ≤ 0.0001 |

Prediction 3 (simulated disagreements are 30–70% of Q.ANT's) already looks very unlikely, given about 5 against Exp015's 300–600. It is formally decided in Part 2.

## What Part 2 will show

`disagreements_sim_vs_qant` tells how much of the Q.ANT mismatch the bfloat16 simulator fails to capture. `disagreements_ref_vs_qant` per arm tells whether QAT changes the real mismatch, even though it could not change the simulated one.
