# PNN-v1 Analysis008 — frequency-basis search

## Result
The frozen Alpha2 topology was tested with five frequency bases using the established TRAIN-only validation protocol.

| Basis | Parameters | Mean validation Q.ANT accuracy | Std |
|---|---:|---:|---:|
| [1,2,3,4] control | 7,778 | 0.725 | 0.04610 |
| [1,2,3] | 5,858 | 0.730 | 0.06403 |
| [1,2] | 3,938 | 0.755 | 0.06103 |
| [1,2,4] | 5,858 | 0.720 | 0.06403 |
| [1,3,5] | 5,858 | 0.680 | 0.07141 |

The preregistered success criterion was not met because [1,2], despite the highest mean validation accuracy, exceeded the allowed stability degradation. Alpha2 therefore remains k=[1,2,3,4].

## Interpretation
The [1,2] result is a useful secondary signal: +0.030 validation accuracy with roughly half the parameters, but greater across-seed variation. It should not be promoted from Exp008. A dedicated frozen confirmation is justified because the candidate was identified by validation and has a strong compactness advantage.

## Conclusion
**Result classification: not_supported_with_promising_secondary_signal.**

No physical photonic hardware performance is measured here.
