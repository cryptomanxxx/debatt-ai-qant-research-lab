# CL-66 predictions — Q.ANT confirmation of the simulator and the grouped-head design (written before any run)

Committed and pushed before any run. Informal predictions, not a formal preregistration.

**Tier 2, owner-approved** ("Kör Q.ANT-bekräftelsen", 2026-11-03). It is one manual dispatch of the existing CL-58 workflow, extended with an experiment choice, with the same permissions as CL-58 (`contents: read`, `packages: read`), in the same Q.ANT research image, on the Q.ANT **CPU backend**. No new secrets and no cost beyond Actions minutes. Physical energy: `NOT_MEASURED`.

## Question

1. **Does the CL-62 simulator flip the same images as Q.ANT,** not only the same number? This is the per-image test that CL-62 to CL-65 could not do.
2. **Does the CL-65 design deliver on Q.ANT what the simulator predicted?** The design is the 28×8 hierarchical model with its head split into 8 calls; the simulator predicted 0.860 accuracy and 46 disagreements per 10k.

## Method

- **Models,** trained in the run with the unchanged local code. Training is deterministic, so the reference accuracies can be compared with the local runs:
  - `mono`: the CL-58 `exact_cosine` grid-4 model (784 → 10), with a single Q.ANT call per logit;
  - `h28x8`: the CL-64/65 28×8 hierarchical model (seeds 11, 22, 33).
- **Evaluation on the Fashion-MNIST test set (10,000 images):**
  - `mono`: one `calc_kan_layer_fprop` + `add_bias_fprop`, as in CL-58.
  - `h28x8_head1`: one call per stage-1 block with bias, then one head call with bias.
  - `h28x8_head8`: the same stage 1, then 8 head calls without bias, summed in float32, then `add_bias_fprop` once.
- **Simulator.** The same paths are evaluated with the CL-62 simulator in the same process.
- **Saved:** the per-image predicted labels from the reference, the simulator and Q.ANT, for every arm and seed.

## Measures

- Disagreements: reference vs Q.ANT (dQ), reference vs simulator (dS), and simulator vs Q.ANT.
- For the flipped sets F_Q = {i : Q.ANT ≠ reference} and F_S = {i : simulator ≠ reference}: the share of F_Q also in F_S (recall), the share of F_S also in F_Q (precision), and the Jaccard index.

## Predictions

1. Reference accuracies equal the local runs exactly:
   - `mono`: 0.8451 / 0.8462 / 0.8078;
   - `h28x8`: 0.8606 / 0.8620 / 0.8567.
2. `mono` Q.ANT disagreements equal CL-58's archived counts (275 / 340 / 381), since the model and the backend are the same.
3. **Main bet (per image):** for `mono`, at least 50% of the images Q.ANT flips are also flipped by the simulator (recall ≥ 0.5, averaged over seeds). The simulator also disagrees with Q.ANT on at most 0.6× as many images as the float reference does.
4. `h28x8_head8` Q.ANT disagreements are within ±30% of the simulator's: 32–59 on average, against the simulated 45.7.
5. `h28x8_head8` Q.ANT accuracy is at least 0.855, averaged over seeds.
6. Q.ANT disagreements fall at least 2.5× from `h28x8_head1` to `h28x8_head8` (simulated: 3.2×).
