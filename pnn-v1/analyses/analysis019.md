# PNN-v1 Analysis019 — SonyAIBORobotSurface1 redundant-readout numerical robustness

## Result

Exp019 preregistered the narrower numerical-robustness signal observed in Exp017–018 and tested it on the frozen SonyAIBORobotSurface1 adaptation from Exp013.

| Metric | Single head | Redundant 3-head |
|---|---:|---:|
| Parameters | 4,674 | 5,702 |
| Mean reference accuracy | 65.97% | 66.07% |
| Mean Q.ANT accuracy | 65.99% | 65.97% |
| Q.ANT accuracy std | 3.60 pp | 2.31 pp |
| Mean reference/Q.ANT disagreements | 1.5 | 1.2 |
| Mean absolute logit error | 0.01644 | 0.01134 |

## Decision

**Classification: supported.**

The preregistered primary criterion required strictly lower mean absolute reference/Q.ANT logit error while retaining Q.ANT accuracy within one percentage point of control. The redundant readout reduced logit MAE by about 31% while Q.ANT accuracy was effectively unchanged.

Disagreements and seed-to-seed accuracy variation also decreased, but these were explicitly secondary non-gating observations.

## Cross-experiment interpretation

Three independent datasets now show lower mean absolute Q.ANT/reference logit error with three-head averaging:

- TwoLeadECG (Exp017): lower logit MAE, higher Q.ANT accuracy, fewer disagreements.
- GunPoint (Exp018): lower logit MAE, higher Q.ANT accuracy, slightly more disagreements from an already near-zero control floor.
- SonyAIBORobotSurface1 (Exp019): lower logit MAE, essentially unchanged Q.ANT accuracy, fewer disagreements.

This supports a narrower PNN-v1 design principle: redundant Q.ANT-native readout averaging can improve numerical agreement between reference and Q.ANT software/simulation evaluation without a demonstrated accuracy penalty across these three tested datasets. It does **not** establish universal disagreement reduction or physical photonic-hardware robustness.

## Next step

The intervention has now earned an explicit model-revision test. Rather than silently modifying Alpha4, define a candidate **Alpha5** that preserves Alpha4's local architecture rule and replaces the single final Q.ANT readout with three independent Q.ANT-native readouts averaged at the logits. The promotion experiment should compare Alpha4 and Alpha5 on the original ECG200 benchmark under a preregistered multi-objective gate.

All measurements use the Q.ANT CPU/software simulation backend.
