# PNN-v1 Analysis009 — compact frequency confirmation

## Result
Exp009 compared the active Alpha2 k=[1,2,3,4] against the frozen compact k=[1,2] candidate. Both used the same Alpha2 topology, all 100 canonical ECG200 TRAIN cases, 10 seeds, and 100 epochs.

| Model | Parameters | Q.ANT correct / 1,000 | Mean Q.ANT accuracy | Std | Logit MAE |
|---|---:|---:|---:|---:|---:|
| Alpha2 k1234 | 7,778 | 834 | 0.834 | 0.02154 | 0.04528 |
| Compact k12 | 3,938 | 852 | 0.852 | 0.01661 | 0.02617 |

The compact candidate produced 18 more correct Q.ANT classifications, lower across-seed variation, and lower reference/Q.ANT logit error while using about 49% fewer parameters.

## Preregistered decision
Both frozen criteria were met. The result records success_criteria_met=true and promotion_decision=promote_compact_k12.

## Conclusion
**Result classification: supported.**

The compact k=[1,2] model is promoted as **PNN-v1 Alpha3**. The result is ECG200-specific and does not establish general superiority. The conventional MLP calibration remains stronger at 0.876 mean accuracy versus Alpha3 0.852, leaving a 0.024 absolute gap.

No physical photonic latency, throughput, energy efficiency, or optical noise is measured.
