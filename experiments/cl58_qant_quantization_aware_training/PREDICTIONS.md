# CL-58 predictions — quantization-aware training for the Q.ANT KAN layer (written before any run)

These predictions were committed and pushed before any run. They are informal predictions, not a formal preregistration.

## Background

- Models trained with an exact cosine and then evaluated on the Q.ANT CPU backend disagree with their float reference on 300–600 of the 10k Fashion-MNIST test images (Exp015, grid 4 and 8).
- Training against a fitted Q.ANT response table did **not** help (Exp015).
- Exp014 found that part of the Q.ANT error is bfloat16 rounding, and the rest is a Q.ANT-specific residual of similar size.

**Question:** does training that sees bfloat16 rounding during training (quantization-aware training with a straight-through estimator), optionally plus small noise, reduce the Q.ANT mismatch?

## Setting

- **Data and model:** Fashion-MNIST, downloaded with the official MD5 sums checked. The single-layer Fourier/KAN model from Exp015 (784 → 10, grid 4 and 8), with Exp015's initialisation.
- **Training:** Adam (lr 1e-3, batch 128) for 3 epochs, seeds 11, 22 and 33. Implemented in NumPy, so it runs both locally and in the Q.ANT image.
- **Arms:**
  - `exact_cosine`: float32 training (the Exp015 baseline).
  - `bf16_ste`: the forward pass during training rounds inputs, phases, amplitudes and cosine outputs to bfloat16 (round-to-nearest-even). Gradients pass straight through the rounding.
  - `bf16_ste_noise`: as `bf16_ste`, plus Gaussian noise (σ = 0.003) on every cosine output during training only. σ is set to the Q.ANT residual RMSE scale in Exp014.
- **Evaluation:**
  - **Part 1 (local):** float32 reference accuracy; accuracy with simulated bfloat16; and how often the two disagree.
  - **Part 2 (GitHub Actions, Q.ANT image):** Q.ANT CPU-backend accuracy via `calc_kan_layer_fprop` + `add_bias_fprop`, as in Exp015. Disagreements are counted against the float reference and against the bfloat16 simulator.

## Predictions

1. `exact_cosine` reproduces Exp015 on Q.ANT: accuracy within 0.01 of 0.830 (grid 4) and 0.821 (grid 8).
2. `exact_cosine` Q.ANT disagreements with the float reference are within ±30% of Exp015 (323 for grid 4, 587 for grid 8).
3. For `exact_cosine`, simulated-bfloat16 disagreements are 30–70% of the real Q.ANT disagreements.
4. Locally, `bf16_ste` cuts simulated-bfloat16 disagreements by ≥ 50% against `exact_cosine`, for both grids.
5. **Main bet:** on real Q.ANT, `bf16_ste` cuts disagreements by ≥ 30% against `exact_cosine` for grid 8, but Q.ANT accuracy improves by less than 0.01.
6. `bf16_ste_noise` cuts Q.ANT disagreements more than `bf16_ste` for grid 8.
7. Float reference accuracy of both QAT arms is within 0.01 of `exact_cosine`.
