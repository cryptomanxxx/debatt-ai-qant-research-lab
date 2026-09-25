# PNN-v1 Analysis005 — local representation width

## Status

PNN-v1 Exp005 completed successfully on the Q.ANT CPU/software-simulation backend. Width selection used the frozen TRAIN/validation boundary from Exp003/004; canonical TEST was evaluated only after selection.

## Observed results

| Candidate | Parameters | Mean validation Q.ANT accuracy | Validation std |
|---|---:|---:|---:|
| alpha1_w4 | 3,890 | 0.685 | 0.05025 |
| alpha1_w6 | 5,834 | 0.670 | 0.09274 |
| alpha1_w8 | 7,778 | 0.725 | 0.04610 |

The preregistered success criterion was met. The selected candidate was alpha1_w8: +0.040 absolute validation Q.ANT accuracy versus w4 while also reducing validation standard deviation.

After selection was frozen, alpha1_w8 achieved 0.819 mean Q.ANT accuracy and 0.818 mean reference accuracy on canonical TEST across 10 seeds. Mean reference/Q.ANT disagreement was 0.5 predictions per 100 test cases.

## Interpretation

Exp005 provides the first preregistered architecture-level improvement over the original alpha1 under the TRAIN-only selection protocol. The evidence is consistent with the hypothesis that the original 8→4 local mapping compressed each eight-sample ECG window too aggressively.

The result is not monotonic with width: w6 performed worse and was much less stable, while w8 performed best. Therefore the evidence supports the tested 8→8 design, not a general rule that wider local representations are always better.

Reference/Q.ANT agreement remains close, so backend substitution is still not the dominant limitation on ECG200.

## Conclusion

**Result classification: supported_with_qualification.**

Supported: alpha1_w8 passed the frozen validation criterion and remained numerically compatible with Q.ANT evaluation.

Qualification: ECG200 is very small, and the selected architecture has not yet been compared under an equal full-TRAIN confirmation protocol against the original alpha1 and conventional MLP baseline.

## Architecture candidate

The selected candidate is promoted provisionally to **PNN-v1 Alpha2 candidate**:

96 input → 12 × Q.ANT Fourier/KAN 8→8 → 96 features → Q.ANT Fourier/KAN 96→2 → 2 classes.

Promotion to the active Alpha2 architecture requires the confirmation experiment below.

## Recommended next experiment

Freeze the architecture and run a confirmation experiment using all 100 canonical ECG200 TRAIN cases. Compare:
1. original Alpha1 (8→4),
2. Alpha2 candidate (8→8),
3. the established MLP-ReLU baseline.

No architecture selection or tuning may use canonical TEST in this confirmation. Report all three models across the same fixed seeds. The purpose is confirmation and calibration against the conventional baseline, not another search.

No result here measures real photonic hardware latency, throughput, energy efficiency or optical noise.
