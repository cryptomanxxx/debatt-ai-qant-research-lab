# PNN-v1 Analysis010 — local-window geometry search

## Result
Exp010 kept Alpha3's compact k=[1,2] basis fixed and changed only the contiguous local-window geometry while preserving 96 concatenated features before the final 96→2 Q.ANT Fourier/KAN head.

| Geometry | Parameters | Mean validation Q.ANT accuracy | Std |
|---|---:|---:|---:|
| 16×6 | 3,170 | 0.770 | 0.05568 |
| 12×8 control | 3,938 | 0.755 | 0.06103 |
| 8×12 | 5,474 | 0.750 | 0.02236 |
| 6×16 | 7,010 | 0.800 | 0.05000 |

The 6×16 candidate exceeded the control by 0.045 absolute validation accuracy and satisfied the preregistered stability gate.

The frozen selected candidate was then evaluated on TEST using the models trained on the 80-case selection subset: mean Q.ANT accuracy 0.863, std 0.01616, reference accuracy 0.863, and zero reference/Q.ANT prediction disagreements.

## Interpretation
The result supports larger local temporal receptive fields as a promising direction in this tested family, but Exp010 does not promote a successor to Alpha3. Its protocol explicitly requires a separate full-TRAIN confirmation.

## Conclusion
**Result classification: supported_candidate_requires_confirmation.**

The 6×16, k=[1,2] candidate advances to a frozen full-TRAIN comparison against active Alpha3. No physical photonic hardware performance is measured.
