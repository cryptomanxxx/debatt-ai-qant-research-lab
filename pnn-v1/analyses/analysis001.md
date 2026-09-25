# PNN-v1 Analysis001 — ECG200 alpha1

## Status

PNN-v1 Exp001 completed successfully on the Q.ANT CPU/software-simulation backend using the canonical ECG200 TRAIN/TEST split.

## Observed results

| Model | Parameters | Mean accuracy | Accuracy std |
|---|---:|---:|---:|
| MLP-ReLU 96→64→32→2 | 8,354 | 0.8767 reference | 0.0047 |
| PNN-v1-alpha1 12×4 g4 | 3,890 | 0.8300 reference / 0.8333 Q.ANT | 0.0356 reference / 0.0340 Q.ANT |

PNN-v1-alpha1 used 4,464 fewer parameters than the MLP baseline (53.44% fewer).

Per-seed Q.ANT accuracies for alpha1 were 0.88, 0.80 and 0.82. The corresponding reference accuracies were 0.88, 0.80 and 0.81.

Numerical compatibility was strong at the classification level: reference and Q.ANT predictions disagreed on 0, 0 and 1 of 100 test cases across the three seeds. Mean absolute logit error averaged 0.03846.

## Interpretation

Exp001 establishes the core feasibility milestone: a compact hierarchical Q.ANT-native Fourier/KAN architecture can be trained on a real public time-series classification benchmark and evaluated through the Q.ANT software backend.

The alpha1 model is substantially smaller than the conventional baseline, but its mean task accuracy is lower and its seed-to-seed variance is much larger. Therefore Exp001 does **not** support a claim that alpha1 outperforms the conventional MLP.

The most actionable weakness is training/initialization stability. Seed 11 reaches 0.88 Q.ANT accuracy, matching the upper range of the MLP runs, while seeds 22 and 33 fall to 0.80 and 0.82. Because the reference model shows essentially the same seed pattern, this instability is primarily present before Q.ANT backend substitution rather than being introduced by reference-vs-Q.ANT numerical mismatch.

The low number of prediction disagreements also suggests that, for this architecture and dataset, improving the learned model itself is currently more important than reducing backend numerical mismatch.

## Conclusion

**Result classification: supported_with_qualification.**

Supported: PNN-v1-alpha1 successfully solves the practical ECG200 task with a compact Q.ANT-native hierarchical architecture and retains close reference/Q.ANT classification agreement.

Qualification: it trails the MLP baseline in mean accuracy and is markedly less stable across seeds.

## Recommended next hypothesis

PNN-v1 Proposal002 should target **stability without abandoning the Q.ANT-native low-fan-in hierarchy**. The experiment should change one controlled architectural/training dimension at a time and evaluate more fixed seeds. It should not yet become a broad architecture search, because ECG200 is small and repeated test-set-guided optimization would create substantial overfitting risk.

No result here measures real photonic hardware latency, throughput, energy efficiency or optical noise.
