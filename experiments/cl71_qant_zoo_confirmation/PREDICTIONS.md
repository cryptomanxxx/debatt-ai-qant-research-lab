# CL-71 predictions — Q.ANT confirmation of the CL-69 model zoo (phase D; written before the run)

Committed and pushed before the Q.ANT run. Informal predictions, not a formal preregistration.

**Tier 2, owner-approved** ("Kör fas D, jag godkänner", 2026-11-03). It is one manual dispatch of `cl58-qant-qat.yml` with a new `cl71` option and the same permissions as CL-58, CL-66 and CL-70 (`contents: read`, `packages: read`), on the Q.ANT **CPU backend**. Physical energy: `NOT_MEASURED`.

## Question

The CL-69 ranking of models by their Q.ANT suitability came from the simulator. **Does the real Q.ANT CPU backend reproduce that ranking, model by model and image by image?**

## Method

- **Models.** They are trained in the run with the unchanged CL-69 code: Fashion-MNIST, Adam lr 1e-3, batch 128, 3 epochs, seeds 11/22/33. The 7 evaluated arms are:
  - two linear models, `mlp_h256x3` and `mlp_sig_h256` (sigmoid);
  - three KAN models: `kan_g8`, `kan_g4`, and `kan_g4_split16` (the same model as `kan_g4`, split into 16 calls plus a bf16 bias);
  - two hybrids: `lin128_kanhead` and `kan32_linhead`.
  - `kan_g4` and `hier_28x8_head8` were already confirmed on Q.ANT in CL-66. `kan_g4` is re-measured here as an anchor; the hierarchical model is not repeated.
- **Q.ANT path.** Every layer goes through `linear_fprop`/`calc_kan_layer_fprop`, `add_bias_fprop`, and then `relu_fprop` or `sigmoid_fprop`. Split KAN calls are summed in float32 before `add_bias_fprop`. The class is the argmax of the logits.
- **Saved.** The per-image reference and Q.ANT labels.
- **Simulator comparison.** The simulator predictions are recomputed locally from the same deterministic training. The Actions reference labels must equal the local ones, which checks that training is identical across machines.

## Simulated values from CL-69 (disagreements per 10k; Q.ANT-path accuracy)

| Model | `mlp_h256x3` | `mlp_sig_h256` | `kan_g8` | `kan_g4` | `kan_g4_split16` | `lin128_kanhead` | `kan32_linhead` |
|---|---|---|---|---|---|---|---|
| Disagreements | 10.0 | 11.3 | 470.7 | 337.0 | 42.7 | 95.0 | 129.7 |
| Accuracy | 0.8670 | 0.8552 | 0.8333 | 0.8284 | 0.8339 | 0.8498 | 0.8393 |

## Predictions

1. The reference labels from Actions equal the local ones for all 18 trained models (6 models × 3 seeds).
2. **Main bet:** for every KAN-containing arm, the mean Q.ANT disagreements are within ±30% of the simulated value. The Spearman rank correlation between simulated and Q.ANT disagreements over all 7 arms is at least 0.9.
3. The linear models have at most 25 disagreements per 10k on Q.ANT (`mlp_sig_h256` at most 40, since the sigmoid path is known to differ slightly).
4. **Per image:** for `kan_g8`, at least 85% of the images Q.ANT flips are also flipped by the simulator (recall ≥ 0.85, averaged over seeds).
5. On Q.ANT, `kan_g8` has the most negative accuracy gap compared with float32, and `mlp_h256x3` has the highest accuracy.
