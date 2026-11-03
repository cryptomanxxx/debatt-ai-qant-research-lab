# CL-68 predictions — does `linear_fprop` accumulate like `calc_kan_layer_fprop`? (written before any run)

Committed and pushed before any run. Informal predictions, not a formal preregistration. This is phase A of the [research goal](../../docs/RESEARCH_ROADMAP.md#research-goal-set-by-the-owner-2026-11-03).

## Question

The accumulation simulator (CL-62 to CL-66) describes Q.ANT's `calc_kan_layer_fprop`: a sequential bfloat16 running sum. **Does Q.ANT's `linear_fprop`, the primitive behind ordinary MLPs, accumulate the same way?** If not, Q.ANT's two main primitives have very different numerical behaviour, and model suitability depends on which primitive a model uses.

## Archived evidence (Q.ANT CPU backend; read before this note was written)

The figures are mean disagreements with the float32 reference per 10k test images, over seeds 11/22/33, with ReLU MLPs, 3 epochs and the PyTorch recipe.

| Archive | Dataset | Mean disagreements by topology |
|---|---|---|
| Exp008 | MNIST | h32_512 1.7, h64_512 3.3, h128_512 2.7, h128 1.3, h256 3.3, h512 2.0 |
| Exp009 | Fashion-MNIST | h64_512 7.7, h128_512 14.0, h128 11.3, h256 12.7, h512 12.7 |

In the same Exp010 run on Fashion-MNIST, a KAN layer with the same 784-term fan-in (grid 1) had 159 disagreements, while ReLU MLPs had 7–11. That already suggests different accumulation, but the models differ, so it is not conclusive.

## Method (Tier 0, local)

- **Models.** NumPy reimplementation of the Exp008/009 MLPs:
  - PyTorch default `nn.Linear` initialisation, U(±1/√fan_in) for weights and biases;
  - Adam (lr 1e-3), batch 128, 3 epochs, seeds 11/22/33;
  - Exp008 topologies on MNIST and Exp009 topologies on Fashion-MNIST.
- **Q.ANT evaluation path, simulated as in the archived code.** Per layer: `linear_fprop`, then `add_bias_fprop`, then `relu_fprop`. After the last layer: `softmax_fprop`, then argmax.
- **Two variants for `linear_fprop`.** Everything else is in bfloat16 in both: inputs, weights, outputs, bias addition and softmax output.
  - `f32acc`: the dot products accumulate in float32.
  - `seq`: each product is rounded to bfloat16 and summed sequentially in bfloat16, as in the KAN simulator, using a C kernel.

## Predictions

1. The NumPy reference accuracies are within ±1.5 pp of the archived means for every topology.
2. **Main bet:** `f32acc` disagreements are within a factor of 2 of the archived Q.ANT means, or within ±3 absolute where the archive is below 5, for at least 9 of the 11 topology–dataset pairs.
3. `seq` disagreements are at least 5× the archived means for all 11 pairs. That would reject sequential bfloat16 accumulation for `linear_fprop`.
4. On Fashion-MNIST, `seq` gives at least 50 disagreements per 10k for every topology.
