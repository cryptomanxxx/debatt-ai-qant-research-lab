# PNN-v1 Analysis003 — residual-head architecture test

## Status

PNN-v1 Exp003 completed successfully on the Q.ANT CPU/software-simulation backend. Candidate selection used a deterministic 20-case validation subset drawn only from the canonical ECG200 TRAIN split. The canonical TEST split was evaluated only after selection was frozen.

## Observed results

| Candidate | Parameters | Mean validation Q.ANT accuracy | Validation std |
|---|---:|---:|---:|
| alpha1 control | 3,890 | 0.685 | 0.05025 |
| alpha2 residual head | 3,986 | 0.680 | 0.05568 |

The preregistered success criterion was not met. Alpha2 neither improved mean validation accuracy nor stability, so alpha1 was selected.

Confirmatory TEST performance for the selected alpha1 was 0.802 mean Q.ANT accuracy with 0.02676 standard deviation across 10 seeds. Mean reference accuracy was 0.805, with 0.5 reference/Q.ANT prediction disagreements per 100 test cases and mean absolute logit error 0.03776.

## Interpretation

The linear residual path around the 48→2 Fourier/KAN head is not supported as the next PNN-v1 architecture. Adding 96 trainable parameters did not improve the validation objective.

The close reference/Q.ANT agreement again indicates that the immediate bottleneck is not backend substitution. Exp003 shifts attention upstream: alpha1's 12 independent local 8→4 blocks may discard useful cross-window temporal structure before the final head sees the signal.

The lower confirmatory TEST mean than Exp002 must not be interpreted as a direct regression: Exp003 trained on 80 canonical TRAIN cases because 20 were reserved for validation, whereas Exp002 trained on all 100.

## Conclusion

**Result classification: not_supported.**

The residual-head hypothesis is falsified under the preregistered Exp003 protocol. Alpha1 remains the active architecture.

## Recommended next hypothesis

Test whether a small second Q.ANT-native hierarchical mixing stage can combine neighboring local representations before classification. This directly targets cross-window temporal structure while preserving low fan-in and avoiding a broad architecture search.

No result here measures real photonic hardware latency, throughput, energy efficiency or optical noise.
