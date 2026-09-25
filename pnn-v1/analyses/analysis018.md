# PNN-v1 Analysis018 — GunPoint cross-dataset redundant readout validation

## Result

Exp018 transferred the Exp017 three-head readout intervention to the frozen GunPoint adaptation from Exp012.

| Metric | Single head | Redundant 3-head |
|---|---:|---:|
| Parameters | 10,514 | 12,822 |
| Mean reference accuracy | 94.93% | 95.27% |
| Mean Q.ANT accuracy | 94.93% | 95.47% |
| Q.ANT accuracy std | 1.24 pp | 1.02 pp |
| Mean reference/Q.ANT disagreements | 0.2 | 0.3 |
| Mean absolute logit error | 0.03331 | 0.02006 |

## Decision

**Classification: not_supported under the preregistered primary criterion.**

The redundant candidate retained and slightly improved Q.ANT accuracy, reduced logit MAE, and reduced seed-to-seed accuracy variation. However, mean prediction disagreements increased from 0.2 to 0.3 rather than decreasing.

## Interpretation

The cross-dataset result does not support a general claim that redundant readout reduces reference/Q.ANT classification disagreements. GunPoint also presents a floor effect: the single-head control averaged only 0.2 disagreements per 150 TEST examples per seed.

Across Exp017 and Exp018, a different signal is consistent: three-head averaging reduced mean absolute Q.ANT/reference logit error on both datasets while also improving mean Q.ANT accuracy and reducing its seed-to-seed variation. This pattern was not the preregistered primary claim in Exp018 and therefore should be treated as a new hypothesis, not a retrospective success.

The next experiment should preregister that narrower hypothesis and test it on a dataset where compatibility error is not already at floor. SonyAIBORobotSurface1 from Exp013 is appropriate: its frozen transfer architecture had materially more reference/Q.ANT disagreements than GunPoint, while avoiding another round of TwoLeadECG-specific tuning.

All measurements use the Q.ANT CPU/software simulation backend. No physical photonic hardware-performance claim is made.
