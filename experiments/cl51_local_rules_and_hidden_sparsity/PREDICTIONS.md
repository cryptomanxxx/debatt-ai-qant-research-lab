# CL-51 predictions (written before any run)

Committed before the experiment was run, so the results can be compared against them. These are informal predictions, not a formal preregistration.

Setting: shared-rule dataset, 30 seeds (101–130), 10 passes per task, final mean test accuracy over T1–T4. For reference from CL-50: backprop 0.999, original Hebbian rule 0.500, frozen random hidden layer 0.541, OR rule 0.882.

## Part A — stabilised local hidden rules (16 → 8 → 2, same readout rule as `local_hybrid`)

| Arm | Prediction |
|---|---|
| `hebb` (original rule) | 0.500, reproducing CL-50 |
| `hebb_nobias` (no bias update) | below 0.65: weights still grow without bound |
| `oja` (Oja's rule on active inputs) | 0.50–0.65: bounded but label-independent features |
| `competitive` (kWTA k=2 plus instar rule) | 0.55–0.75 |
| `dfa` (direct feedback alignment) | above 0.90: carries label information |

## Part B — hidden-layer kWTA with backprop

- H=8: k=8 ≈ 0.999; k=4 ≥ 0.95; k=2 clearly lower; k=1 at or below the OR rule.
- H=32: k=4 ≥ 0.95, with fewer readout weight terms than dense H=8.
