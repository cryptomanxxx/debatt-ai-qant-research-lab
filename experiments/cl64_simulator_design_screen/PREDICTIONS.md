# CL-64 predictions — screening Q.ANT designs with the accumulation simulator (written before any run)

Committed and pushed before any run. Informal predictions, not a formal preregistration.

## Question

The sequential-bfloat16 simulator from CL-62 predicted archived Q.ANT CPU-backend results in CL-63. **Used as a local design tool, which Fourier/KAN design gives the best trade-off between accuracy and Q.ANT mismatch, and does the simulator also reproduce the archived Exp027 hierarchical results?**

## Designs (Fashion-MNIST, grid 4, Adam lr 1e-3, batch 128, 3 epochs, seeds 11/22/33)

1. **Grouped monolithic.**
   - The CL-58 `exact_cosine` grid-4 models (784 → 10) are evaluated with the logit sum split into g contiguous Q.ANT calls, g ∈ {1, 2, 4, 8, 16, 49, 98, 196, 784}.
   - Group outputs are summed in float32, as Exp022 did. Unlike Exp022, a bfloat16 bias is added at the end.
   - The model and its reference accuracy are the same for every g; only the evaluation changes.
2. **Hierarchical (Exp027).**
   - Each of g ∈ {8, 16, 28} blocks maps 784/g pixels to w ∈ {2, 8} features, and a head maps the g·w features to 10.
   - Both stages are Fourier/KAN layers with bias, each evaluated as one Q.ANT call (`calc_kan_layer_fprop` + `add_bias_fprop`).
   - The models are reimplemented and trained in NumPy, following Exp027's initialisation and training recipe. Torch is not installed, so these are not Exp027's models.

**Simulation.** Each Q.ANT call is simulated with `seq` (CL-62): bfloat16 operands, cosines and products, a bfloat16 running sum and a bfloat16 bias. The reference is the float32 forward pass.

**Diagnostic, not a prediction:** for the hierarchical designs, simulate one stage at a time with the other exact, to locate the error.

## Derived before running

- **Grouping.** If the per-call error is a random walk of rounding errors whose size scales with the running sum, the total logit error after g calls falls about like 1/√g. Disagreements should fall at least that fast.
- **Exp022 check.** Its archived Q.ANT counts fall faster: 323 → 114 → 38 for g = 1, 4, 16, a log–log slope of about −0.77.
- **Hierarchical designs.** Archived Q.ANT disagreements are nearly flat at 110–138 per 10k, although stage-1 fan-in varies from 112 to 392 terms. That suggests the error is not set by stage-1 fan-in.

## Predictions

1. NumPy reference accuracies are within ±1.5 pp of the archived Exp027 reference accuracy for each of the 6 hierarchical designs (means over seeds).
2. **Main bet (out of sample):** simulated disagreements are within ±30% of the archived Exp027 Q.ANT disagreements for at least 5 of the 6 designs.
3. Across the 6 hierarchical designs, simulated disagreements vary by at most a factor of 1.5 (the archive varies by 1.25×).
4. Grouped monolithic disagreements fall monotonically with g. The log–log slope over g = 1–49 is between −1.0 and −0.5.
5. At g = 784, `seq` disagreements are within 2× of float32-accumulation disagreements, the operand- and output-rounding floor.
6. Grouped monolithic with g ≥ 16 has fewer simulated disagreements than every hierarchical design. The best hierarchical design (28×8) has a higher simulated Q.ANT-path accuracy than the grouped monolithic model at any g.
