# CL-70 predictions — fan-in budget of Q.ANT's linear and KAN primitives, plus the activations (written before the Q.ANT run)

Committed and pushed before the Q.ANT run. Informal predictions, not a formal preregistration. This is CL-70 (simulator budget curve, Tier 0) together with phase B of the [research goal](../../docs/RESEARCH_ROADMAP.md#research-goal-set-by-the-owner-2026-11-03).

**Phase B is Tier 2, owner-approved** ("Jag godkänner", 2026-11-03): one manual dispatch of `cl58-qant-qat.yml` with `experiment: cl70`, with the same permissions as CL-58/CL-66 (`contents: read`, `packages: read`), on the Q.ANT **CPU backend**. Physical energy: `NOT_MEASURED`.

## Question

**How large a fan-in does each Q.ANT primitive tolerate before its output error grows, and do the activation primitives behave as the simulator assumes?**

## Method

- **Synthetic layers.** Deterministic, seeded per condition, 3 seeds, 256 samples, 10 outputs, with inputs uniform on [−1, 1].
  - `linear_fprop`: fan-in n ∈ {1, 4, …, 65,536}, with weights N(0, 1/√n).
  - `calc_kan_layer_fprop`: grid 4, n_in ∈ {1, 4, …, 16,384}, so 4 to 65,536 terms, with amplitudes N(0, 1/√terms), as in Exp021.
- **Error measure.** Relative MAE: the mean absolute error against a float64 computation with the float32 parameters, divided by the mean absolute exact output.
- **Simulator (run locally before this note).** [`results_simulator.json`](results_simulator.json). The two models are float32 accumulation (`f32acc`) and the sequential bfloat16 sum (`seq`), using the CL-67/68 C kernels.
- **Q.ANT.** The same layers, measured on the CPU backend.
- **Activations.** `relu_fprop`, `sigmoid_fprop`, `softmax_fprop` and `add_bias_fprop` are compared element by element with the formulas the simulator assumes: max(x, 0), bfloat16(float32 sigmoid), bfloat16(float32 softmax) and bfloat16(a + b).

## Simulator values (relative MAE, means over 3 seeds)

| Terms | 1 | 4 | 16 | 64 | 256 | 1,024 | 4,096 | 16,384 | 65,536 |
|---|---|---|---|---|---|---|---|---|---|
| linear, `f32acc` | 0.22% | 0.25% | 0.27% | 0.28% | 0.28% | 0.27% | 0.27% | 0.28% | 0.28% |
| linear, `seq` | 0.22% | 0.34% | 0.52% | 0.89% | 1.7% | 3.3% | 6.5% | 12.7% | 23.5% |
| KAN, `seq` | — | 0.43% | 0.60% | 0.92% | 1.7% | 3.2% | 6.4% | 12.6% | 24.4% |

For the KAN layer the simulated log–log slope from 256 to 65,536 terms is 0.48, close to the √n random-walk value of 0.5.

## Predictions

1. `linear_fprop` relative MAE is within ±30% of the `f32acc` value at all 9 fan-ins, staying at about 0.3% up to 65,536. At n ≥ 1,024 it is at most a third of the `seq` value.
2. **Main bet:** `calc_kan_layer_fprop` relative MAE is within ±20% of the simulated `seq` value for every size from 256 to 65,536 terms. The largest sizes, 16,384 and 65,536 terms, are beyond anything tested before (Exp021 stopped at 6,272). At ≤ 64 terms, Q.ANT may be up to 35% higher (the CL-63 residual).
3. The KAN log–log slope over 256–65,536 terms is between 0.40 and 0.55.
4. **Budget:**
   - the KAN layer exceeds 1% relative error above about 100 terms per call, and 10% above about 10,000;
   - `linear_fprop` stays below 0.5% up to 65,536.
5. **Activations:**
   - `relu_fprop` matches max(x, 0) exactly, 100%;
   - `add_bias_fprop` matches bfloat16(a + b) for at least 99.9% of elements;
   - `sigmoid_fprop` matches bfloat16(float32 sigmoid) for at least 99%;
   - `softmax_fprop` matches bfloat16(float32 softmax) for at least 95%, and its argmax equals the argmax of the logits for at least 99.5% of rows.
