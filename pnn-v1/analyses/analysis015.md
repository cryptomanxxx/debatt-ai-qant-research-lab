# PNN-v1 Analysis015 — decision-margin robustness intervention

## Result

Exp015 compared the frozen TwoLeadECG PNN under ordinary cross-entropy training with a preregistered training-only margin penalty (target margin 1.0, lambda 0.1).

| Metric | Control | Margin-regularized |
|---|---:|---:|
| Mean Q.ANT TEST accuracy | 82.985% | 83.108% |
| Q.ANT accuracy std | 2.364 pp | 2.163 pp |
| Mean reference accuracy | 83.055% | 83.213% |
| Mean ref/Q.ANT disagreements | 4.2 | 4.2 |
| Mean absolute logit error | 0.02082 | 0.02043 |
| Median reference margin across seeds | 2.0900 | 2.0216 |

## Decision

**Classification: not_supported.**

The preregistered criterion required strictly fewer mean reference/Q.ANT disagreements while retaining Q.ANT accuracy within one percentage point of control. Accuracy was retained and slightly improved, but disagreements did not decrease at all.

The intervention also did not enlarge the reported median TEST decision margin; it moved slightly downward. Therefore Exp014's strong association between low margins and disagreement should not be interpreted as evidence that this simple training penalty can fix the compatibility problem.

## Interpretation

Exp014 localized where disagreements occur; Exp015 shows that a generic true-class training-margin penalty is not sufficient to remove them. The next experiment should diagnose *where in the frozen network the Q.ANT/reference perturbation accumulates* on disagreement cases rather than immediately trying another loss coefficient.

A useful next decomposition is local blocks versus final head: compute reference and Q.ANT local representations, then separate (1) local representation perturbation, (2) propagation of that perturbed representation through the reference head, and (3) the head's own Q.ANT evaluation error.

All measurements use the Q.ANT CPU/software simulation backend. No physical photonic hardware-performance claim is made.
