# CL-67 predictions — training through the simulated Q.ANT accumulation (written before any run)

Committed and pushed before any run. Informal predictions, not a formal preregistration.

## Question

[CL-66](../cl66_qant_confirmation/README.md) confirmed on Q.ANT that the CPU backend behaves like sequential bfloat16 accumulation per logit. Its errors, not operand rounding, cause the mismatch, which is why CL-58's operand-only QAT failed. **If a model is trained with the forward pass going through the simulated sequential-bfloat16 accumulation, does it get fewer mismatches and a higher accuracy on the Q.ANT path, with one Q.ANT call per logit?**

## Method (Tier 0, local, simulator only)

- **Model and recipe.** The CL-58 monolithic Fourier/KAN layer (784 → 10, grid 4), Fashion-MNIST, Adam lr 1e-3, batch 128, 3 epochs, seeds 11/22/33.
- **Arms:**
  - `exact`: the CL-58 `exact_cosine` training, unchanged.
  - `seq_ste`: every training forward pass uses the CL-62 simulator (bfloat16 operands, cosines and products, a sequential bfloat16 running sum and a bfloat16 bias). Backward uses straight-through gradients: the float32 gradient formulas, evaluated at the bfloat16 operands and cosines.
  - `finetune`: 2 epochs `exact`, then 1 epoch `seq_ste`.
- **Speed.** A small C kernel does the sequential bfloat16 sum. It is checked bit for bit against the NumPy simulator (`cl62.accumulate(..., "seq")`).
- **Evaluation** on the 10,000 test images:
  - reference accuracy, from the float32 forward pass;
  - Q.ANT-path accuracy, from the simulator with one call per logit;
  - disagreements between the two.

## Derived before running

- **What training cannot change.** Training cannot remove the rounding: every logit is still a 3,136-term sequential bfloat16 sum.
- **What it can change.** It can enlarge the decision margins of the images that rounding flips. It can also shrink the running-sum magnitudes, on which each rounding error scales.
- **Expectation.** The reduction should therefore be partial, not down to the 16-call floor of about 43 disagreements (CL-64).

## Predictions

1. The C kernel is bit-identical to the NumPy simulator.
2. `exact` reproduces CL-58/CL-62 exactly: reference accuracy 0.8451 / 0.8462 / 0.8078, and simulated disagreements 279 / 340 / 392.
3. `seq_ste` reduces the mean disagreements by at least 25% compared with `exact` (337.0), to ≤ 253.
4. **Main bet:** the `seq_ste` Q.ANT-path accuracy is at least 0.3 pp above `exact`'s (0.8284), so ≥ 0.8314.
5. `seq_ste` stays above 100 disagreements: training does not replace splitting the sum.
6. `finetune` achieves at least half of `seq_ste`'s reduction in disagreements.
