# CL-71 — Q.ANT confirmation of the model zoo (phase D)

**Tier 2, owner-approved** ("Kör fas D, jag godkänner", 2026-11-03). It was one dispatch of [`cl58-qant-qat.yml`](../../.github/workflows/cl58-qant-qat.yml) with `experiment: cl71` ([run 41734752823](https://github.com/cryptomanxxx/debatt-ai-qant-research-lab/actions/runs/41734752823), 6 minutes) on the Q.ANT **CPU backend**, with the same permissions as CL-58, CL-66 and CL-70. This is phase D of the [research goal](../../docs/RESEARCH_ROADMAP.md#research-goal-set-by-the-owner-2026-11-03). [Predictions](PREDICTIONS.md) were pushed before the run (commit `941f4c1`). Physical energy: `NOT_MEASURED`.

- [`results_qant.json`](results_qant.json) holds the per-image reference and Q.ANT labels for 7 arms × 3 seeds. It was rebuilt from the job log, and the raw bytes match the logged SHA-256. The artifact is `cl71-results-qant`, with zip SHA-256 `ab464083…`.
- [`results.json`](results.json) is the local analysis (`--mode analyze`). It retrains the same models, checks the reference labels, and adds the simulator labels from the CL-69 path.

## Question

**Does the real Q.ANT CPU backend reproduce the CL-69 simulator ranking of model suitability, model by model and image by image?**

## Results

Means over 3 seeds, out of 10,000 Fashion-MNIST test images. The float32 accuracy is the CPU/GPU reference. Recall is the share of Q.ANT's flipped images that the simulator also flips; precision is the share of the simulator's flipped images that Q.ANT also flips.

| Arm | Float32 accuracy | Q.ANT accuracy | Gap (pp) | Q.ANT disagreements | Simulated | Simulator vs Q.ANT | Recall / precision |
|---|---|---|---|---|---|---|---|
| `mlp_h256x3` | 0.8670 | **0.8669** | −0.01 | **11.7** | 10.0 | 4.3 | 0.75 / 0.88 |
| `mlp_sig_h256` | 0.8547 | 0.8549 | +0.02 | 12.3 | 11.3 | 4.3 | 0.78 / 0.85 |
| `kan_g4_split16` | 0.8330 | 0.8336 | +0.06 | 44.0 | 42.7 | 21.3 | 0.74 / 0.77 |
| `lin128_kanhead` | 0.8499 | 0.8500 | +0.01 | 97.3 | 95.0 | 26.3 | 0.85 / 0.87 |
| `kan32_linhead` | 0.8393 | 0.8393 | 0.00 | 127.0 | 129.7 | 30.0 | 0.89 / 0.86 |
| `kan_g4` | 0.8330 | 0.8285 | −0.45 | 332.0 | 337.0 | 62.0 | 0.92 / 0.90 |
| **`kan_g8`** | 0.8429 | 0.8334 | **−0.95** | **462.0** | 470.7 | 105.7 | 0.90 / 0.88 |

- The reference labels are identical between Actions (NumPy 2.3.5) and the local run (NumPy 2.5.3) for all 18 trained models.
- The Spearman rank correlation between simulated and Q.ANT disagreements over the 7 arms is **1.0**.
- `kan_g4` reproduces CL-66's Q.ANT count exactly (332.0).

## Predictions

| # | Prediction | Outcome |
|---|---|---|
| 1 | Reference labels from Actions equal the local ones for all 18 models | ✓ all 18 identical |
| 2 | **Every KAN-containing arm within ±30% of the simulator; Spearman ≥ 0.9** | ✓ 0.98–1.03×; Spearman 1.0 |
| 3 | Linear models ≤ 25 disagreements (sigmoid ≤ 40) | ✓ 11.7 and 12.3 |
| 4 | `kan_g8` recall ≥ 0.85 | ✓ 0.90 |
| 5 | `kan_g8` has the most negative gap; `mlp_h256x3` the highest Q.ANT accuracy | ✓ −0.95 pp; 0.8669 |

**5 of 5 held.**

## Findings

1. **The simulator's ranking of model suitability holds on the Q.ANT CPU backend.** Every arm is within 3% of its simulated disagreement count, and the rank order is identical (ρ = 1.0).
   - Where many images flip (the KAN models), 85–92% of Q.ANT's flipped images are the same ones the simulator flips.
   - Where few flip, as in the linear models and the split KAN, the remaining flips are near-ties and per-image agreement drops to 74–78%.
2. **Answer to the research question for numerical fidelity, now measured on Q.ANT:**
   - **Best suited:** `linear_fprop` models. A 3-layer ReLU MLP keeps its float32 accuracy (0.867, a gap of −0.01 pp) with about 12 disagreements per 10k. A sigmoid MLP behaves the same, despite the one-step sigmoid rounding difference found in CL-70.
   - **Worst suited:** large monolithic KAN layers. `kan_g8` loses 0.95 pp, with 462 disagreements per 10k.
   - **In between:** KAN made Q.ANT-friendly by short sums. Splitting into 16 calls or using a KAN head gives 44–97 disagreements with no accuracy loss.
   - **Why:** `calc_kan_layer_fprop` rounds its running sum to bfloat16 after every term (CL-62 to CL-66, CL-70), while `linear_fprop` accumulates in float32 or better (CL-70).

## What this does and does not show

It is a CPU-backend result on one dataset, with 3 seeds and 7 architectures. It says nothing about the photonic hardware: speed, energy, analog noise and physical matrix size remain phase E.
