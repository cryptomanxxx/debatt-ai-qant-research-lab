# CL-63 — Out-of-sample test of the Q.ANT accumulation simulator

**Tier 0** under [docs/RISK_TIERS.md](../../docs/RISK_TIERS.md). It runs locally in about 1 minute and makes no Q.ANT run. It uses the archived Q.ANT CPU-backend results of Exp021 and Exp022, and reuses the CL-58 Fashion-MNIST download. **Exploratory.** [Predictions](PREDICTIONS.md) were pushed before the run (commit `c8e82a5`). Physical energy: `NOT_MEASURED`.

Reproduce from the repository root: `python -m experiments.cl63_simulator_out_of_sample.run`, which writes [`results.json`](results.json).

## Question

[CL-62](../cl62_qant_accumulation_simulator/README.md) found that rounding the running sum to bfloat16 after every term (`seq`) reproduces the CL-58 Q.ANT disagreement counts. **Does the same simulator, unchanged, predict earlier Q.ANT results that played no part in choosing it?**

## Method

- **A. Exp021 scaling law.**
  - The 108 synthetic conditions (1–784 inputs × 1–8 components × 3 seeds, 256 samples) are regenerated from Exp021's seeded generator.
  - Simulated mean absolute error (MAE) against Exp021's float32 cosine reference is compared with the archived Q.ANT MAE.
- **B. Exp022 grouped accumulation.**
  - Exp022 evaluated one Q.ANT call per contiguous group of 784/g pixels and summed the group outputs in float32.
  - In the code that ran (commit `0b2947c`), the bias call sits inside a comment, so the archived Q.ANT logits have no bias and no final rounding. The simulation follows that path; a version with a bfloat16 bias is reported as a sensitivity check. (Codex caught this. The first version of this PR added the bias, and its numbers were slightly different: 337/116/43.)
  - This is simulated with `seq` inside each group on the CL-58 `exact_cosine` grid-4 models (3 seeds).
  - These are not Exp022's torch models (torch is not installed), only the same architecture and recipe, so B is an aggregate comparison.
- **Simulator variants,** all with bfloat16 operands, cosines, products and output:
  - `f32acc`: float32 sum.
  - `seq`: pixel-major order. `seq_kmajor`: component-major order.
  - `pair`: pairwise bfloat16.
  - `block32`: float32 blocks of 32 terms.

## Results

**A. Exp021** (36 conditions averaged over seeds; ratio = simulated MAE / Q.ANT MAE)

| Variant | Log–log exponent | Conditions within ±15% | Median ratio | r(log MAE), 108 points |
|---|---|---|---|---|
| **Q.ANT (archived)** | **0.364** | — | — | — |
| `f32acc` | 0.044 | 0% | 0.24 | 0.53 |
| **`seq`** | **0.397** | **83%** | **0.97** | **0.996** |
| `seq_kmajor` | 0.395 | 83% | 0.95 | 0.996 |
| `pair` | 0.101 | 0% | 0.40 | 0.87 |
| `block32` | 0.053 | 0% | 0.30 | 0.63 |

- The `seq` ratio rises from 0.67–0.92 at 1–32 terms to 0.98–1.00 at 256–6,272 terms.
- The six conditions outside ±15% all have at most 32 terms, and `seq` underestimates in all of them.

**B. Exp022** (mean disagreements per 10k against the float reference)

| Groups | Q.ANT (Exp022, archived) | `seq` simulated | `f32acc` simulated | `seq` with bias (sensitivity check) |
|---|---|---|---|---|
| 1 (monolithic) | 323.0 | 337.3 (1.04×) | 18.0 | 337.0 |
| 4 | 114.3 | 114.3 (1.00×) | 8.3 | 116.3 |
| 16 | 37.7 | 41.0 (1.09×) | 8.0 | 42.7 |

## Predictions

| # | Prediction | Outcome |
|---|---|---|
| 1 | Regenerated inputs match Exp021's setup | – not directly checkable: the archive stores only error summaries, not the inputs. The generator code is identical. |
| 2 | `seq` within ±15% in ≥ 80% of conditions | ✓ 83% |
| 3 | `seq` exponent within 0.364 ± 0.05 | ✓ 0.397 |
| 4 | `f32acc` ≥ 3× too low at 784 × 8, within ±30% at 1 term | ✓ 14× too low, and −24% |
| 5 | `pair` and `block32` exponents < 0.25 | ✓ 0.10 and 0.05 |
| 6 | **Grouped `seq` within ±25% of Exp022 for 4 and 16 groups** | ✓ 0% and +9% |

**5 of 5 testable predictions held.**

## Findings

1. **The simulator predicts results it was never fitted to.**
   - **Exp021:** it reproduces Q.ANT's error at every scale from 128 to 6,272 terms within 0–14%. It also reproduces the scaling exponent (0.40 against 0.36), with r = 0.996 on log MAE over 108 points.
   - **Exp022:** it predicts the 3× and 9× drops in disagreements from splitting the sum into 4 and 16 Q.ANT calls, within 0–9%.
   - **Every other scheme fails:** float32, pairwise and blockwise accumulation are off by 2.5–14×.
2. **The √n mechanism holds.** Q.ANT's own slope approaches the 0.5 random-walk exponent at large n, as derived before the run.
3. **There is a small extra Q.ANT error at low term counts.** With 1–32 terms the simulator underestimates Q.ANT by up to 33%. The source could be the cosine implementation or another operand detail, matching the Q.ANT-specific residual in Exp014. It is negligible at the fan-ins used in practice.
4. **The summation order cannot be resolved by these data:** pixel-major and component-major give the same MAE within 3%.
5. **The practical consequence is a cheap local proxy for the Q.ANT CPU backend's numerical behaviour.** Exp021–030 explored low-fan-in and hierarchical designs on Q.ANT run by run; this proxy allows such designs to be screened, and trained against, locally.

## What this does and does not show

It is strong aggregate evidence across two independent earlier experiments and a model that was fixed in advance. It is still not a per-image comparison; that needs a new Q.ANT run that saves predictions (Tier 2). Exp022's models were not retrained here. Everything concerns the CPU backend, not photonic hardware.
