# CL-69 predictions — a systematic model zoo for Q.ANT numerical fidelity (written before any run)

Committed and pushed before any run. Informal predictions, not a formal preregistration. This is phase C of the [research goal](../../docs/RESEARCH_ROADMAP.md#research-goal-set-by-the-owner-2026-11-03).

## Question

**Which model types and properties keep their CPU/GPU accuracy best, and which lose it worst, on the Q.ANT CPU-backend path, and why?** Each property is varied one at a time from a base model.

## Setting (Tier 0, local, simulator only)

- **Data and training.** Fashion-MNIST, reusing the CL-58 download. Adam lr 1e-3, batch 128, 3 epochs, seeds 11/22/33, all in NumPy.
- **Primitive models.** They follow the evidence so far:
  - `calc_kan_layer_fprop`: bfloat16 operands, cosines and products, and a sequential bfloat16 running sum (CL-62 to CL-66, confirmed on Q.ANT per image).
  - `linear_fprop`: bfloat16 inputs, weights and outputs, with float32 accumulation (CL-68, aggregate match only).
  - `add_bias_fprop`: bfloat16.
  - `relu_fprop` is exact on bfloat16. For `sigmoid_fprop` and `softmax_fprop`, my assumption (not yet checked against Q.ANT) is a float32 computation with bfloat16 output.
- **Models, varied one property at a time from the bases** (mlp_h256 for the linear family, kan_g4 for the KAN family):

| Family | Models | Property varied |
|---|---|---|
| Linear | `logreg` (784→10), `mlp_h64`, `mlp_h256` (base), `mlp_h1024` | width |
| Linear | `mlp_h256x2`, `mlp_h256x3` | depth |
| Linear | `mlp_sig_h256` | activation (sigmoid) |
| KAN | `kan_g1`, `kan_g2`, `kan_g4` (base), `kan_g8` | grid, meaning terms per call (784·g) |
| KAN | `kan_g4_split16` (same model as `kan_g4`, logit split into 16 calls) | calls per logit |
| KAN | `hier_28x8_head8` (CL-65 design) | hierarchy |
| Hybrid | `lin128_kanhead` (linear 784→128, ReLU, KAN 128→10 grid 4) | KAN only in the head |
| Hybrid | `kan32_linhead` (KAN 784→32 grid 4, linear 32→10) | KAN only in the first layer |

- **Per model:**
  - float32 ("CPU/GPU") accuracy;
  - simulated Q.ANT-path accuracy;
  - the accuracy gap (Q.ANT − float);
  - disagreements per 10k;
  - Q.ANT calls and the longest sequential bfloat16 sum per call.

## Predictions

1. Every linear-family model has fewer than 20 disagreements per 10k, and an accuracy gap within ±0.1 pp.
2. Width (h64 to h1024) and depth (1 to 3 hidden layers) change linear-family disagreements by at most 2× from the base.
3. KAN monolithic disagreements rise with grid: g1 < g2 < g4 < g8.
4. **Main bet: the worst-suited model is `kan_g8`,** with the most disagreements and the most negative accuracy gap. The best-suited models, those with the fewest disagreements, are linear-family models.
5. `lin128_kanhead` has at least 3× fewer disagreements than `kan_g4`, because its KAN sum has 512 terms instead of 3,136. `kan32_linhead` keeps at least half of `kan_g4`'s disagreements, because its sums still have 3,136 terms.
6. Across all KAN-containing models, disagreements increase with the longest sequential bfloat16 sum per call: Spearman rank correlation ≥ 0.7.
