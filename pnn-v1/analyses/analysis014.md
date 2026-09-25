# PNN-v1 Analysis014 — TwoLeadECG disagreement localization

## Result

Exp014 kept the Exp013 TwoLeadECG model and training protocol frozen and pooled diagnostics over ten seeds.

- Total predictions: **11,390**
- Reference/Q.ANT agreements: **11,348**
- Disagreements: **42**
- Mean Q.ANT accuracy: **82.99%**
- Mean disagreements: **4.2 per TEST set**

The separation around the decision boundary was large:

| Diagnostic | Agreement | Disagreement |
|---|---:|---:|
| Median reference top1-top2 margin | 2.01948 | **0.01668** |
| Median max-logit-perturbation / margin | 0.01402 | **1.73903** |

## Decision

**Classification: supported.**

The preregistered criterion required disagreement cases to have both a lower median reference margin and a higher median perturbation-to-margin ratio than agreement cases. Both conditions were satisfied by a wide margin.

The result supports a margin-sensitivity mechanism: relatively modest reference/Q.ANT numerical perturbations become classification-changing primarily when the reference model is already close to its decision boundary.

This does not establish that low margin is the only compatibility mechanism, nor that increasing margins will necessarily improve Q.ANT accuracy. That intervention must be tested separately.

## Next hypothesis

Test a frozen training-only margin intervention on TwoLeadECG. The architecture, k=[1,2], data crop, optimizer, epochs, and seeds remain unchanged. Compare the ordinary cross-entropy control with a preregistered cross-entropy plus margin penalty intended to enlarge the top1/top2 training-logit gap. Evaluate whether it reduces reference/Q.ANT disagreements without materially reducing Q.ANT TEST accuracy.

All Q.ANT measurements use the CPU/software simulation backend. No physical photonic hardware-performance claim is made.
