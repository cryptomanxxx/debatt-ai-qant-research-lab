# CL-63 predictions — does the CL-62 simulator predict earlier Q.ANT results it was never fitted to? (written before any run)

Committed and pushed before any run. Informal predictions, not a formal preregistration.

## Question

[CL-62](../cl62_qant_accumulation_simulator/README.md) found that rounding the running sum to bfloat16 after every term (`seq`) reproduces the CL-58 Q.ANT disagreement counts. **Does the same simulator, unchanged, predict two earlier Q.ANT CPU-backend results that played no part in choosing it?**

## Tests (Tier 0: local, no Q.ANT run)

**A. Exp021 accumulation scaling law.** The archive is [`results/exp021_qant_accumulation_scaling_law.json`](../../results/exp021_qant_accumulation_scaling_law.json).

- **Data.** Synthetic KAN layers with 1–784 inputs, 1–8 components, 10 outputs and 256 samples, 3 seeds, so 108 conditions in all. The Q.ANT mean absolute error (MAE) against an exact float32 cosine is archived for each condition.
- **Exact regeneration.** The inputs come from `np.random.default_rng(seed*100000 + ni*100 + nk)`, so they can be regenerated exactly. The regeneration is checked against the run's own float32 reference, which it should match bit for bit, since both are computed in NumPy.
- **Variants:**
  - `f32acc`: bfloat16 operands, float32 sum and bfloat16 output (the CL-58 simulator).
  - `seq`: the CL-62 simulator (pixel-major order).
  - `seq_kmajor`: the same, but with component-major order.
  - `pair`: pairwise bfloat16 accumulation.
  - `block32`: float32 blocks of 32 terms.
- **Comparison.** Each variant's MAE is compared with the Q.ANT MAE per condition, and its log–log scaling exponent with Q.ANT's (0.364).

**B. Exp022 grouped accumulation.** The archive is [`results/exp022_qant_grouped_accumulation_architecture.json`](../../results/exp022_qant_grouped_accumulation_architecture.json).

- **What Exp022 did.** It split each Fourier/KAN logit into 4 or 16 contiguous input groups. Each group was evaluated by a separate Q.ANT call, the group outputs were summed in float32, and a bfloat16 bias was added.
- **What Q.ANT gave** (disagreements per 10k against the float reference, means of 3 seeds): monolithic 323.0, 4 groups 114.3, 16 groups 37.7.
- **Models.** The torch models of Exp022 cannot be retrained here, because torch is not installed. Instead I use the CL-58 `exact_cosine` grid-4 models, which have the same architecture, initialisation scale, optimiser, batch size and epoch count (3 seeds). They are different trained models, so this is an aggregate comparison.
- **Simulation.** Exp022's grouping is reproduced with `seq` inside each group.

## Derived before running

- **Random-walk scaling.** With output scale held roughly constant, as Exp021 normalised it, each bfloat16 addition adds a rounding error of about u·|S|. These errors add up like a random walk, so the accumulated error grows like √n.
- **Expected exponent.** The large-n exponent should therefore approach 0.5. It should be lower over the full range 1–6,272, because operand rounding sets a floor at small n.
- **Check against Q.ANT.** Q.ANT's own local slope from 3,136 to 6,272 terms is log(0.0445/0.0326)/log 2 = 0.45. That is consistent with this mechanism before any simulation.

## Predictions

1. Regenerated Exp021 inputs reproduce the archived condition setup. The float32 reference used here matches the definition in `experiments/021_qant_accumulation_scaling_law/run.py`.
2. `seq` is within ±15% of the Q.ANT MAE in at least 80% of the 36 (inputs × components) conditions, averaged over seeds.
3. The `seq` log–log exponent is within 0.364 ± 0.05.
4. `f32acc` underestimates the Q.ANT MAE by at least 3× at 784 × 8 terms, and is within ±30% at 1 term.
5. `pair` and `block32` have exponents below 0.25.
6. **Main bet (B):** `seq` with grouping gives mean disagreements within ±25% of the Exp022 Q.ANT values for both 4 groups (114.3) and 16 groups (37.7).
