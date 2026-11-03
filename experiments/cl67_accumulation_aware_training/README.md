# CL-67 — Training through the simulated Q.ANT accumulation

**Tier 0** under [docs/RISK_TIERS.md](../../docs/RISK_TIERS.md). It runs locally in about 2 minutes on 4 processes, using the simulator only; no Q.ANT run was made. **Exploratory.** [Predictions](PREDICTIONS.md) were pushed before the run (commit `6660983`). Physical energy: `NOT_MEASURED`.

Reproduce from the repository root: `python -m experiments.cl67_accumulation_aware_training.run`, which writes [`results.json`](results.json).

## Question

[CL-66](../cl66_qant_confirmation/README.md) confirmed on Q.ANT that the CPU backend behaves like sequential bfloat16 accumulation per logit. **If the model is trained with its forward pass going through that accumulation, does it get fewer mismatches and a higher accuracy on the Q.ANT path, still with one call per logit?**

## Method

- **Model and recipe.** The CL-58 monolithic Fourier/KAN layer (784 → 10, grid 4), Fashion-MNIST, Adam lr 1e-3, batch 128, 3 epochs, seeds 11/22/33.
- **Arms:**
  - `exact`: the CL-58 training, unchanged.
  - `seq_ste`: every training forward pass goes through the CL-62 simulator, with straight-through gradients.
  - `finetune`: 2 `exact` epochs, then 1 `seq_ste` epoch.
- **C kernel.** The sequential bfloat16 sum runs in a C kernel ([`seq_bf16.c`](seq_bf16.c)), tested bit-identical to the NumPy simulator. A `seq_ste` training run takes about 1.7 minutes, against 0.5 minutes for `exact`.
- **Evaluation.** Each model is evaluated with the simulator, one call per logit. CL-66 confirmed this simulator on Q.ANT.

## Results

Means over 3 seeds (per-seed values are in `results.json`).

| Arm | Reference accuracy | Q.ANT-path accuracy (simulated) | Disagreements per 10k |
|---|---|---|---|
| `exact` | 0.8330 | 0.8284 | 337.0 |
| `seq_ste` | 0.8314 | 0.8306 (+0.22 pp) | 313.3 (−7%) |
| `finetune` | 0.8335 | 0.8314 (+0.30 pp) | 320.3 (−5%) |
| *For comparison: `exact` split into 16 calls (CL-64)* | 0.8330 | 0.8339 (+0.55 pp) | 42.7 (−87%) |

- **Per seed, Q.ANT-path accuracy compared with `exact`:**
  - `seq_ste`: +0.40, +0.29 and −0.04 pp;
  - `finetune`: +0.55, +0.32 and +0.03 pp.
- **Disagreements** fall for every seed in both arms.

## Predictions

| # | Prediction | Outcome |
|---|---|---|
| 1 | The C kernel is bit-identical to the NumPy simulator | ✓ (`tests/test_cl67_accumulation_aware_training.py`) |
| 2 | `exact` reproduces CL-58/CL-62 exactly | ✓ 0.8451 / 0.8462 / 0.8078; 279 / 340 / 392 |
| 3 | `seq_ste` reduces disagreements by ≥ 25% | ✗ −7% |
| 4 | **`seq_ste` Q.ANT-path accuracy ≥ +0.3 pp** | ✗ +0.22 pp (`finetune` reaches +0.30 pp) |
| 5 | `seq_ste` stays above 100 disagreements | ✓ 313 |
| 6 | `finetune` achieves ≥ half of `seq_ste`'s reduction | ✓ 70% |

**3 of 6 held.**

## Findings

1. **Training through the accumulation helps only a little.**
   - Q.ANT-path accuracy rises by 0.2–0.3 pp, and disagreements fall by 5–7%.
   - The gain is not consistent across seeds: one seed shows no improvement.
   - A single fine-tuning epoch does about as well as training through the simulator for all three epochs.
2. **Architecture beats training here.** Splitting the same logit into 16 Q.ANT calls (CL-64) gives twice the accuracy gain and 12× the reduction in disagreements, with no retraining. This matches the derivation made before the run: every logit is still a 3,136-term sequential bfloat16 sum, and training can only move margins, not remove the rounding.
3. **Practical rule for the Q.ANT CPU backend:** keep each call's sequential sum short (tens to about a hundred terms), and treat accumulation-aware training as an optional final touch.

## What this does and does not show

All results are simulated (the simulator was confirmed on Q.ANT in CL-66). There are 3 seeds and one recipe. A stronger training-side method, such as margin losses, rounding noise matched to the error model, or longer training, might do more. That was not tried.
