# PNN-v1 Analysis017 — redundant Q.ANT readout robustness

## Result

Exp017 compared the frozen TwoLeadECG architecture with a single Q.ANT Fourier/KAN readout head against three independently parameterized Q.ANT-compatible heads whose logits were averaged.

| Metric | Single head | Redundant 3-head |
|---|---:|---:|
| Parameters | 5,842 | 7,126 |
| Mean reference accuracy | 83.06% | 83.70% |
| Mean Q.ANT accuracy | 82.99% | 83.85% |
| Q.ANT accuracy std | 2.36 pp | 1.54 pp |
| Mean reference/Q.ANT disagreements | 4.2 | 3.3 |
| Mean absolute logit error | 0.02082 | 0.01419 |

## Decision

**Classification: supported.**

The preregistered criterion required strictly fewer reference/Q.ANT prediction disagreements and Q.ANT accuracy no more than one percentage point below control. The redundant readout reduced mean disagreements from 4.2 to 3.3 while increasing mean Q.ANT accuracy.

The redundant candidate also reduced mean absolute logit error and seed-to-seed Q.ANT accuracy variation. These were not required for success but are consistent with the robustness hypothesis.

## Interpretation

This is the first positive intervention following the Exp014–016 diagnostic sequence. The result supports redundant Q.ANT-native readout as a candidate compatibility design principle on TwoLeadECG, but it is still a single-dataset intervention result.

The preregistered next step is cross-dataset validation without target-specific architecture search. GunPoint is suitable because the frozen Alpha4 design already transferred strongly there in Exp012. The next experiment should reproduce the Exp012 GunPoint adaptation and change only the number of readout heads.

All measurements use the Q.ANT CPU/software simulation backend. No physical photonic hardware-performance claim is made.
