# CL-70 — Fan-in budget of Q.ANT's linear and KAN primitives, plus the activations

- **Simulator part:** Tier 0, local.
- **Q.ANT part:** phase B of the [research goal](../../docs/RESEARCH_ROADMAP.md#research-goal-set-by-the-owner-2026-11-03), **Tier 2, owner-approved** ("Jag godkänner", 2026-11-03). It was one dispatch of [`cl58-qant-qat.yml`](../../.github/workflows/cl58-qant-qat.yml) with `experiment: cl70` ([run 41730263785](https://github.com/cryptomanxxx/debatt-ai-qant-research-lab/actions/runs/41730263785), 25 seconds of measurement) on the Q.ANT **CPU backend**, with the same permissions as CL-58 and CL-66.

[Predictions](PREDICTIONS.md), with the simulator values, were pushed before the Q.ANT run (commit `854fb62`). Physical energy: `NOT_MEASURED`.

- [`results_simulator.json`](results_simulator.json) is the local run.
- [`results_qant.json`](results_qant.json) was transcribed from the job log's `SUMMARY_JSON` line. The per-seed values reproduce the logged means to 1e-15. The run's artifact is `cl70-results-qant`, with zip SHA-256 `23199146…`.
- **Rerun for softmax.** The first run's batched `softmax_fprop` call was rejected. After Codex review, softmax was called one row at a time, and the same approved measurement was dispatched again ([run 41732943458](https://github.com/cryptomanxxx/debatt-ai-qant-research-lab/actions/runs/41732943458); artifact zip SHA-256 `e1b73540…`). All 51 linear and KAN values and every other activation value were **identical** to the first run.

## Question

**How large a fan-in does each Q.ANT primitive tolerate, and do the activation primitives behave as the simulator assumes?**

## Method

- **Synthetic layers.** Seeded per condition, 3 seeds, 256 samples, 10 outputs, inputs uniform on [−1, 1].
  - `linear_fprop` at fan-in 1–65,536.
  - `calc_kan_layer_fprop` at grid 4, with 4–65,536 terms. Exp021 had only reached 6,272.
- **Error measure.** Relative MAE: the mean absolute error against float64 with the float32 parameters, divided by the mean absolute exact output.
- **Activations.** Compared element by element with the formulas the simulator assumes.

## Results

Relative MAE, means over 3 seeds; the ratio is Q.ANT divided by the matching simulator model.

| Terms | 1 | 4 | 16 | 64 | 256 | 1,024 | 4,096 | 16,384 | 65,536 |
|---|---|---|---|---|---|---|---|---|---|
| **`linear_fprop`, Q.ANT** | 0.22% | 0.29% | 0.31% | 0.32% | 0.32% | 0.32% | 0.32% | 0.32% | 0.32% |
| linear, simulated float32 accumulation (ratio) | 0.22% (1.00) | 0.25% (1.16) | 0.27% (1.15) | 0.28% (1.16) | 0.28% (1.18) | 0.27% (1.17) | 0.27% (1.18) | 0.28% (1.18) | 0.28% (1.17) |
| linear, simulated sequential bf16 | 0.22% | 0.34% | 0.52% | 0.89% | 1.7% | 3.3% | 6.5% | 12.7% | 23.5% |
| **`calc_kan_layer_fprop`, Q.ANT** | — | 0.55% | 0.72% | 1.00% | 1.75% | 3.27% | 6.41% | 12.5% | 24.3% |
| KAN, simulated sequential bf16 (ratio) | — | 0.43% (1.28) | 0.60% (1.20) | 0.92% (1.09) | 1.71% (1.02) | 3.24% (1.01) | 6.38% (1.00) | 12.6% (0.99) | 24.4% (0.99) |

The Q.ANT KAN log–log slope from 256 to 65,536 terms is 0.47.

**Activations on Q.ANT:**

| Primitive | Compared with | Exact matches | Largest difference |
|---|---|---|---|
| `relu_fprop` | max(x, 0) | 100% | 0 |
| `add_bias_fprop` | bf16(a + b) | 100% | 0 |
| `sigmoid_fprop` | bf16(float32 sigmoid) | **69.7%** | 0.0039 (one bf16 step near 0.5–1) |
| `softmax_fprop` (one row per call) | bf16(float32 softmax) | **62.0%** | 0.0039 (one bf16 step) |
| `softmax_fprop` argmax | argmax of the logits | 100% (500 rows) | — |

## Predictions

| # | Prediction | Outcome |
|---|---|---|
| 1 | `linear_fprop` within ±30% of the float32 model at all fan-ins, and ≤ ⅓ of the sequential-bf16 value from 1,024 up | ✓ 1.00–1.18×; 0.32% against 3.3–23.5% |
| 2 | **KAN within ±20% of the sequential-bf16 simulator from 256 to 65,536 terms (≤ 35% above at ≤ 64)** | ✓ within 2% from 256 to 65,536; +9% to +28% at ≤ 64 |
| 3 | KAN slope from 256 to 65,536 terms between 0.40 and 0.55 | ✓ 0.47 |
| 4 | **Budget:** KAN above 1% beyond about 100 terms and above 10% beyond about 10,000; linear below 0.5% up to 65,536 | ✓ KAN reaches 1.0% at 64 and 12.5% at 16,384; linear stays at 0.32% |
| 5 | Activations: relu 100%, bias ≥ 99.9%, sigmoid ≥ 99%, softmax ≥ 95%, softmax argmax ≥ 99.5% | ✗ relu, bias and softmax argmax hold (100%); sigmoid matches only 69.7%, softmax only 62.0% |

**4 of 5 held.**

## Findings

1. **`linear_fprop` tolerates any fan-in tested.** Its error is flat at 0.32% from 16 to 65,536 terms, so it does not grow with the length of the sum. That is float32 (or better) accumulation; the remaining error is bfloat16 rounding of the operands and output.
   - It is 1.17× the simple float32 model, a constant factor that is probably a detail of the output or operand rounding.
   - At 65,536 terms, sequential bfloat16 accumulation would be 73× worse.
2. **`calc_kan_layer_fprop` follows the simulator to within 2% from 256 up to 65,536 terms,** ten times beyond the earlier Exp021 range. Its error grows like √n (slope 0.47), passing 1% at about 64 terms and 10% at about 10,000.
   - Below about 64 terms there is a small extra Q.ANT error (+9–28%), as CL-63 saw.
3. **The fan-in budget per Q.ANT call, for a target relative error:**

   | Target error | `linear_fprop` | `calc_kan_layer_fprop` |
   |---|---|---|
   | 1% | any (≥ 65,536) | about 64 terms |
   | 3% | any | about 1,000 terms |
   | 10% | any | about 10,000 terms |

4. **`sigmoid_fprop` is not bf16(float32 sigmoid):** 30% of its outputs differ by one bfloat16 step. That is too small to matter much for classification. CL-69's sigmoid MLP had 11 disagreements under the assumed path, but that row is now known to rest on a slightly wrong activation model.
5. **`softmax_fprop` only accepts single rows,** consistent with the archived per-sample use in Exp001/003/009.
   - Like sigmoid, its outputs differ from bf16(float32 softmax) by one bf16 step in 38% of values.
   - Its argmax always equals the argmax of the logits, so it never changes a classification.
   - The Q.ANT transcendental activations evidently use a slightly different approximation or rounding than float32 followed by RNE to bf16.

## What this does and does not show

It measures the **CPU backend** on synthetic layers. It says nothing about the photonic hardware, where matrix size, analog noise, speed and energy are unknown. The "fidelity versus fan-in" half of the research hypothesis is now mapped for both primitives on the CPU backend. The "speed versus fan-in" half needs hardware (phase E).
