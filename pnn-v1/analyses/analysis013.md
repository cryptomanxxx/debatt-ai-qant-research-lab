# PNN-v1 Analysis013 — multi-dataset transfer benchmark

## Result

Exp013 tested the frozen Alpha4 principle on three additional preregistered UCR/UEA datasets without target-TEST architecture search.

| Dataset | Mean Q.ANT accuracy | Std | Mean ref/Q.ANT disagreements | MLP calibration |
|---|---:|---:|---:|---:|
| ItalyPowerDemand | 95.99% | 0.78 pp | 0.9 | 96.83% |
| SonyAIBORobotSurface1 | 65.99% | 3.60 pp | 1.5 | 64.36% |
| TwoLeadECG | 82.99% | 2.36 pp | 4.2 | 88.63% |

The preregistered accuracy condition was met on 2/3 datasets, but the compatibility condition required mean reference/Q.ANT prediction disagreements <=1.0 on every dataset. SonyAIBORobotSurface1 and TwoLeadECG exceeded that threshold.

## Decision

**Classification: not_supported.**

The Alpha4 principle is not broadly validated under the preregistered criterion. This is informative rather than a reason to alter the result: accuracy transfer and reference/Q.ANT compatibility separate on the new datasets.

ItalyPowerDemand provides a strong positive transfer case. SonyAIBORobotSurface1 is primarily an accuracy/generalization failure under this frozen protocol, while TwoLeadECG is especially useful diagnostically because its Q.ANT accuracy remains above 0.80 but its disagreement count rises to 4.2 per TEST set on average.

## Next hypothesis

The next experiment should localize the TwoLeadECG disagreement mechanism before changing the architecture. In particular, measure whether reference/Q.ANT disagreements concentrate at low classification margins and whether logit error/margin predicts disagreement. This preserves the frozen model and turns the failed compatibility gate into a diagnostic experiment.

All Q.ANT measurements use the CPU/software simulation backend. No physical photonic hardware-performance claim is made.
