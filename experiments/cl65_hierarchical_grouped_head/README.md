# CL-65 — Hierarchical Fourier/KAN with a grouped head

**Tier 0** under [docs/RISK_TIERS.md](../../docs/RISK_TIERS.md). It runs locally in about 2 minutes on 4 processes, using the simulator only; no Q.ANT run was made. **Exploratory.** [Predictions](PREDICTIONS.md) were pushed before the run (commit `5a161ee`). Physical energy: `NOT_MEASURED`.

Reproduce from the repository root: `python -m experiments.cl65_hierarchical_grouped_head.run`, which writes [`results.json`](results.json).

## Question

In [CL-64](../cl64_simulator_design_screen/README.md), the most accurate hierarchical design, 28×8, kept 148 simulated Q.ANT disagreements per 10k, mostly from its head's 896-term sequential sum. **If the head is split into h Q.ANT calls summed in float32, does the design keep its accuracy and approach the stage-1 floor?**

## Method

- **Models.** The 28×8 and 16×8 designs, trained with the unchanged CL-64 code (seeds 11/22/33, Fashion-MNIST, grid 4, 3 epochs). The reference results reproduced CL-64 exactly.
- **Head.** The g·w head features are split into h contiguous groups. Each group is one simulated Q.ANT call (sequential bfloat16, [CL-62](../cl62_qant_accumulation_simulator/README.md)). The group outputs are summed in float32, and a bfloat16 bias is added once.
- **Diagnostics.** "Head only" means stage 1 exact with the grouped head simulated. The "floor" means stage 1 simulated with the head exact.

## Results

Means over 3 seeds; disagreements are out of 10,000 test images, against the float32 forward pass.

| Design, head calls | Q.ANT calls | Terms per head call | Simulated accuracy | Disagreements | Head only | Stage-1 floor |
|---|---|---|---|---|---|---|
| 28×8, h=1 (CL-64) | 29 | 896 | 0.8580 | 147.7 | 135.0 | 35.3 |
| 28×8, h=2 | 30 | 448 | 0.8594 | 86.7 | 78.3 | 35.3 |
| 28×8, h=4 | 32 | 224 | 0.8592 | 59.7 | 49.0 | 35.3 |
| **28×8, h=8** | **36** | **112** | **0.8600** | **45.7** | 28.7 | 35.3 |
| 28×8, h=16 | 44 | 56 | 0.8596 | 40.7 | 23.0 | 35.3 |
| 28×8, h=28 | 56 | 32 | 0.8596 | 40.3 | 17.3 | 35.3 |
| 16×8, h=1 (CL-64) | 17 | 512 | 0.8535 | 120.0 | 103.0 | 62.7 |
| 16×8, h=8 | 24 | 64 | 0.8544 | 62.0 | 26.7 | 62.7 |
| 16×8, h=16 | 32 | 32 | 0.8540 | 62.7 | 17.7 | 62.7 |

Reference accuracy is 0.8598 for 28×8 and 0.8542 for 16×8 at every h. For comparison, the CL-64 grouped monolithic model with 16 calls has a simulated accuracy of 0.8339 and 42.7 disagreements.

## Predictions

| # | Prediction | Outcome |
|---|---|---|
| 1 | Reference accuracy and h=1 results reproduce CL-64 exactly | ✓ |
| 2 | Disagreements fall monotonically with h for both designs | ✗ 28×8 ✓; 16×8 rises from 62.0 to 62.7 at h=16 |
| 3 | **28×8 at h=28: ≤ 53 disagreements and accuracy ≥ 0.855** | ✓ 40.3 and 0.8596 |
| 4 | That design is ≥ 2 pp more accurate than grouped monolithic g=16, with ≤ 1.5× its disagreements | ✓ +2.6 pp, and 40.3 against 42.7 |
| 5 | 16×8 at h=16: ≤ 1.5× its stage-1 floor | ✓ 62.7 against 62.7 |

**4 of 5 held.**

## Findings

1. **Splitting the head removes the hierarchical designs' extra mismatch without costing accuracy.**
   - For 28×8, 8 head calls (36 Q.ANT calls in total) cut disagreements from 148 to 46, and the simulated accuracy stays at 0.860.
   - Beyond about 8 head calls, the head contributes little, and stage 1 sets the floor: about 35 for 28×8 and about 63 for 16×8.
2. **According to the simulator, this is the best design tested so far.** 28×8 with a grouped head is 2.6 pp more accurate than the best grouped monolithic model (0.860 against 0.834) at the same or lower mismatch (40–46 against 43).
3. **The remaining mismatch is in stage 1.** Wider blocks with fewer pixels, such as 28 blocks of 28 pixels, have the lower floor. Making stage 1 even smaller would trade more calls for less mismatch.

## What this does and does not show

All results are simulated. They rest on a simulator that matched four archived Q.ANT experiments in aggregate (CL-62 to CL-64), but none of these designs has run on Q.ANT. The natural confirmation is one Q.ANT CPU-backend run of 28×8 with 8 head calls that saves per-image predictions (Tier 2). That would test the design and, at the same time, the simulator image by image.
