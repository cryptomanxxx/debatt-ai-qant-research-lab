# PNN-v1 Analysis016 — stagewise Q.ANT perturbation decomposition

## Result

Exp016 decomposed the frozen TwoLeadECG model's reference/Q.ANT difference into local-representation perturbation, propagation through the reference head, and the head's own Q.ANT evaluation error.

Across 11,390 TEST predictions there were 11,348 agreements and 42 disagreements.

| Median metric | Agreement | Disagreement |
|---|---:|---:|
| Local representation MAE | 0.005165 | 0.005121 |
| Propagation-only logit perturbation | 0.016020 | 0.019333 |
| Isolated head Q.ANT perturbation | 0.019774 | 0.018504 |
| End-to-end logit perturbation | 0.026658 | 0.027000 |

## Decision

**Classification: not_supported.**

The preregistered hypothesis required disagreement cases to have both higher local-representation MAE than agreement cases and propagation-only perturbation larger than isolated-head Q.ANT perturbation. The first condition failed: local representation MAE was essentially unchanged and slightly lower for disagreement cases. The second condition held only narrowly.

## Interpretation

No single early stage dominates the disagreement mechanism. Together with Exp014, the evidence is more consistent with small distributed numerical perturbations becoming decision-changing when a sample lies close to the classifier's decision boundary.

Exp015 showed that a generic training-margin penalty did not reduce disagreements. The next useful intervention is therefore architectural rather than another loss coefficient: create a more numerically robust decision readout while keeping the learned local Q.ANT representation fixed in structure.

All measurements use the Q.ANT CPU/software simulation backend. No physical photonic hardware-performance claim is made.
