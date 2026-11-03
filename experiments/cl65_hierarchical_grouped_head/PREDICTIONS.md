# CL-65 predictions — hierarchical Fourier/KAN with a grouped head (written before any run)

Committed and pushed before any run. Informal predictions, not a formal preregistration.

## Question

In [CL-64](../cl64_simulator_design_screen/README.md), the most accurate hierarchical design, 28×8, reached a reference accuracy of 0.860. Its simulated Q.ANT mismatch was 147.7 disagreements per 10k, and the head's 896-term sequential sum caused most of it: 135.0 with the head alone simulated, against 35.3 with stage 1 alone. **If the head is split into several Q.ANT calls whose outputs are summed in float32, does the design keep its accuracy and approach the stage-1 floor?**

## Method (Tier 0, local, simulator only)

- **Models.** The 28×8 and 16×8 hierarchical designs, trained exactly as in CL-64: the same code, seeds 11/22/33, Fashion-MNIST, grid 4, 3 epochs.
- **Head grouping.** At evaluation, the head's g·w input features (224 or 128) are split into h contiguous groups, with h ∈ {1, 2, 4, 8, 16, 28} for 28×8 and h ∈ {1, 2, 4, 8, 16} for 16×8.
  - Each group is one simulated Q.ANT call without bias.
  - Group outputs are summed in float32, then a bfloat16 bias is added once.
  - Stage 1 is unchanged: one simulated call per block.
- **Diagnostics.** Stage 1 exact with the grouped head simulated, and stage 1 simulated with the head exact. The second is the stage-1 floor.
- **Measured.** Disagreements with the float32 forward pass, and accuracy on the simulated Q.ANT path.

## Predictions

1. Training is deterministic: the reference accuracies and the h = 1 simulated disagreements equal CL-64's exactly (28×8: 0.8598 and 147.7; 16×8: 0.8542 and 120.0).
2. Disagreements fall monotonically with h for both designs.
3. **Main bet:** for 28×8 at h = 28 (32 terms per head call), disagreements are at most 1.5× the stage-1 floor (about 35, so ≤ 53), and the simulated Q.ANT-path accuracy is at least 0.855.
4. That design is at least 2 pp more accurate than the CL-64 grouped monolithic model with 16 calls (0.834). Its disagreements are at most 1.5× that model's (43, so ≤ 64).
5. For 16×8 at h = 16, disagreements are at most 1.5× its stage-1 floor (about 63, so ≤ 94).
