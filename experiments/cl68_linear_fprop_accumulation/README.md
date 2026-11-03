# CL-68 — Does `linear_fprop` accumulate like `calc_kan_layer_fprop`?

**Tier 0** under [docs/RISK_TIERS.md](../../docs/RISK_TIERS.md). It runs locally in about 2 minutes on 4 processes and makes no Q.ANT run. It is compared with the archived Q.ANT CPU-backend results of Exp008 (MNIST) and Exp009 (Fashion-MNIST). This is phase A of the [research goal](../../docs/RESEARCH_ROADMAP.md#research-goal-set-by-the-owner-2026-11-03). **Exploratory.** [Predictions](PREDICTIONS.md) were pushed before the run (commit `0ac6e08`). Physical energy: `NOT_MEASURED`.

Reproduce from the repository root: `python -m experiments.cl68_linear_fprop_accumulation.run`, which writes [`results.json`](results.json).

## Question

The accumulation simulator (CL-62 to CL-66) describes Q.ANT's KAN primitive `calc_kan_layer_fprop` as a sequential bfloat16 running sum. **Does `linear_fprop`, the primitive behind ordinary MLPs, accumulate the same way?**

## Method

- **Models.** NumPy reimplementations of the archived ReLU MLPs:
  - 6 topologies on MNIST (Exp008) and 5 on Fashion-MNIST (Exp009);
  - PyTorch default initialisation, Adam (lr 1e-3), batch 128, 3 epochs, seeds 11/22/33.
- **Evaluation path.** Each model is evaluated along the archived Q.ANT path: `linear_fprop` → `add_bias_fprop` → `relu_fprop` …, with `softmax_fprop` and argmax at the end. Everything is in bfloat16, and two `linear_fprop` variants are compared:
  - `f32acc`: float32 accumulation;
  - `seq`: bfloat16 products and a sequential bfloat16 sum, using a C kernel.
- **Comparison.** Disagreements with the float32 reference, per 10,000 test images, against the archived Q.ANT means.

## Results

Means over 3 seeds; disagreements are per 10,000 test images.

| Dataset | Topology | Reference accuracy, NumPy / archive | Q.ANT (archive) | `f32acc` | `seq` |
|---|---|---|---|---|---|
| MNIST | h32_512 | 0.9610 / 0.9609 | 1.7 | 3.0 | 19.0 |
| MNIST | h64_512 | 0.9669 / 0.9671 | 3.3 | 3.0 | 14.7 |
| MNIST | h128_512 | 0.9705 / 0.9726 | 2.7 | 4.0 | 15.3 |
| MNIST | h128 | 0.9637 / 0.9630 | 1.3 | 2.3 | 12.7 |
| MNIST | h256 | 0.9698 / 0.9700 | 3.3 | 2.3 | 10.0 |
| MNIST | h512 | 0.9742 / 0.9737 | 2.0 | 2.0 | 11.0 |
| Fashion | h64_512 | 0.8601 / 0.8587 | 7.7 | 11.3 | 64.3 |
| Fashion | h128_512 | 0.8589 / 0.8590 | 14.0 | 9.7 | 50.0 |
| Fashion | h128 | 0.8530 / 0.8503 | 11.3 | 10.0 | 53.0 |
| Fashion | h256 | 0.8559 / 0.8587 | 12.7 | 8.7 | 47.7 |
| Fashion | h512 | 0.8615 / 0.8650 | 12.7 | 10.7 | 51.7 |
| **Sum, MNIST** | | | **14.3** | **16.7 (1.16×)** | 82.7 (5.8×) |
| **Sum, Fashion** | | | **58.3** | **50.3 (0.86×)** | 266.7 (4.6×) |

## Predictions

| # | Prediction | Outcome |
|---|---|---|
| 1 | Reference accuracies within ±1.5 pp of the archive | ✓ within 0.35 pp for all 11 |
| 2 | **`f32acc` within 2× (or ±3 absolute below 5) of the archive for ≥ 9 of 11** | ✓ 11 of 11 |
| 3 | `seq` ≥ 5× the archive for all 11 | ✗ 5 of 11 (range 3.0–11.4×) |
| 4 | Fashion `seq` ≥ 50 for every topology | ✗ h256 gives 47.7 |

**2 of 4 held.** The two misses were in the size of `seq`'s overshoot, which I overestimated. Its direction is clear in all 11 pairs: `seq` is above the archive every time, by 3.0× or more.

## Findings

1. **`linear_fprop` behaves like float32 accumulation, not like the KAN primitive.**
   - Float32 accumulation reproduces the archived Q.ANT disagreements for all 11 topology–dataset pairs; summed per dataset, it is within 0.86–1.16×.
   - Sequential bfloat16 accumulation overshoots every pair, by 4.6–5.8× in sum.
   - So Q.ANT's two main primitives differ numerically. The KAN layer rounds its running sum to bfloat16 (CL-62 to CL-66); the linear layer does not.
2. **For the research goal, the primitive a model uses matters more than its size, at least on the CPU backend.**
   - A ReLU MLP keeps about 1–14 disagreements per 10k at a fan-in of 784.
   - A KAN layer with the same fan-in has 124–390 (Exp010, CL-66).
   - The MLP's mismatch hardly depends on width (128–512) or depth (one or two hidden layers).
3. **Why `seq` overshoots less than predicted** (an observation, not tested): many MNIST and Fashion-MNIST pixels and ReLU outputs are exactly zero. Their products add nothing, so the effective sequential sum is shorter than the fan-in.

## What this does and does not show

It is an aggregate match on reimplemented models, with small archived counts, so single topologies are noisy. It is not a per-image check and not a direct measurement of `linear_fprop`. It concerns the **CPU backend**: whether the photonic hardware's matrix operations behave like float32 accumulation is unknown, since that is analog optics. A direct check would be phase B: one Q.ANT run measuring `linear_fprop`'s error against fan-in, as Exp021 did for the KAN layer (Tier 2).
