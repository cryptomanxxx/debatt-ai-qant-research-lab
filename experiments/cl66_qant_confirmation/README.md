# CL-66 — Q.ANT confirmation: per-image test of the simulator and the grouped-head design

**Tier 2, owner-approved** ("Kör Q.ANT-bekräftelsen", 2026-11-03). One manual dispatch of [`cl58-qant-qat.yml`](../../.github/workflows/cl58-qant-qat.yml) with `experiment: cl66` ([run 41708172554](https://github.com/cryptomanxxx/debatt-ai-qant-research-lab/actions/runs/41708172554), 3 minutes) on the **Q.ANT CPU backend**, with the same permissions as CL-58. [Predictions](PREDICTIONS.md) were pushed before the run (commit `0f189c6`). Physical energy: `NOT_MEASURED`. These are CPU-backend results, not photonic hardware results.

[`results_qant.json`](results_qant.json) was rebuilt from the job log. The raw per-image prediction bytes match the logged SHA-256 (`397b4fe1…`). The run's artifact is `cl66-results-qant`, and its zip SHA-256 is `961f5d1d…`.

## Question

1. Does the CL-62 simulator (sequential bfloat16 accumulation) flip the **same images** as Q.ANT?
2. Does the CL-65 design, 28×8 hierarchical with its head split into 8 calls, deliver on Q.ANT what the simulator predicted?

## Method

- **Models,** trained in the run with the unchanged local code:
  - `mono`: the CL-58 grid-4 model, with one Q.ANT call per logit;
  - `h28x8`: the CL-64/65 28×8 hierarchical model, seeds 11/22/33.
- **Evaluation.** Each model is evaluated on Q.ANT (`calc_kan_layer_fprop` + `add_bias_fprop`) and with the simulator along identical paths, on all 10,000 Fashion-MNIST test images.
  - `head1` uses one head call.
  - `head8` uses 8 head calls summed in float32, with the bias added once.
- **Flipped sets.** F = the images where a backend's prediction differs from the float32 reference.
  - Recall = |F_Q ∩ F_S| / |F_Q|, precision = |F_Q ∩ F_S| / |F_S|.
  - Chance overlap = |F_Q|·|F_S| / 10,000.

## Results

Means over 3 seeds, out of 10,000 test images.

| Arm | Q.ANT calls | Reference accuracy | Q.ANT accuracy | Simulator accuracy | Ref vs Q.ANT | Ref vs simulator | Simulator vs Q.ANT | Flipped by both (chance) | Recall / precision |
|---|---|---|---|---|---|---|---|---|---|
| `mono` | 1 | 0.8330 | 0.8285 | 0.8284 | 332.0 | 337.0 | **62.0** | **305.0** (11.4) | **0.92 / 0.90** |
| `h28x8_head1` | 29 | 0.8598 | 0.8584 | 0.8580 | 156.0 | 147.7 | 73.3 | 116.0 (2.3) | 0.74 / 0.78 |
| **`h28x8_head8`** | **36** | 0.8598 | **0.8598** | 0.8600 | **45.7** | 45.7 | 33.3 | 29.0 (0.2) | 0.63 / 0.63 |

Per seed, `mono` Q.ANT disagreements are 275 / 340 / 381, identical to CL-58's archive. For `h28x8_head8` they are 38 / 51 / 48.

## Predictions

| # | Prediction | Outcome |
|---|---|---|
| 1 | Reference accuracies equal the local runs exactly | ✓ all six |
| 2 | `mono` Q.ANT disagreements equal CL-58 (275 / 340 / 381) | ✓ identical |
| 3 | **Per image: recall ≥ 0.5, and simulator vs Q.ANT ≤ 0.6× reference vs Q.ANT** | ✓ recall 0.92; 62 against 332 (0.19×) |
| 4 | `h28x8_head8` Q.ANT disagreements 32–59 (simulated 45.7) | ✓ 45.7 |
| 5 | `h28x8_head8` Q.ANT accuracy ≥ 0.855 | ✓ 0.8598 |
| 6 | Q.ANT disagreements fall ≥ 2.5× from `head1` to `head8` | ✓ 3.4× (156 → 46) |

**6 of 6 held.**

## Findings

1. **The simulator flips the same images as Q.ANT, not just the same number.**
   - For the monolithic model, 92% of the images Q.ANT flips are also flipped by the simulator, and 90% of the simulator's flips are Q.ANT flips. Chance would give about 3%.
   - The simulator disagrees with Q.ANT on 62 images, against 332 for the float32 reference, so it removes 81% of the gap.
   - This confirms the CL-62 mechanism image by image: the Q.ANT CPU backend behaves like sequential bfloat16 accumulation per logit.
2. **The CL-65 design works on Q.ANT as predicted.**
   - 28×8 with an 8-call head reaches a Q.ANT accuracy of 0.8598, the same as its float32 reference, with 45.7 disagreements. The simulator predicted 45.7 exactly in the mean.
   - Compared with CL-58's monolithic Q.ANT models, that is 3.0 pp more accurate (0.8598 against 0.8285) and 7.3× fewer disagreements (46 against 332). It costs 36 calls instead of 1.
3. **At low mismatch, the per-image match weakens** (recall 0.63 for `head8`). The remaining flips sit on near-ties, where details the simulator does not reproduce exactly start to matter, such as summation order or the small Q.ANT residual seen at few terms (CL-63). The counts still agree.

## What this does and does not show

It confirms the simulator and the design on the **Q.ANT CPU backend**, for Fourier/KAN layers on Fashion-MNIST, with 3 seeds. It says nothing about photonic hardware, energy, or other Q.ANT functions. The simulator is not bit-exact.
