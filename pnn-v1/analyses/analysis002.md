# PNN-v1 Analysis002 — alpha1 stabilization

## Status

PNN-v1 Exp002 completed successfully on the Q.ANT CPU/software-simulation backend using the canonical ECG200 TRAIN/TEST split and 10 fixed seeds.

## Observed results

| Condition | Parameters | Mean Q.ANT accuracy | Q.ANT accuracy std | Mean ref/Q.ANT disagreements | Mean logit MAE |
|---|---:|---:|---:|---:|---:|
| alpha1 raw control | 3,890 | 0.824 | 0.03169 | 0.3 / 100 | 0.03713 |
| alpha1 per-case z-score | 3,890 | 0.819 | 0.02948 | 0.5 / 100 | 0.03763 |

The preregistered success criterion was technically met: z-score normalization reduced Q.ANT accuracy standard deviation while keeping the mean Q.ANT accuracy loss within the frozen 0.01 absolute tolerance.

## Interpretation

The stabilization effect is small. Standard deviation fell by about 0.00221 absolute (3.17 to 2.95 percentage points), while mean Q.ANT accuracy fell by 0.005 absolute (82.4% to 81.9%). This is evidence for a modest variance reduction, not evidence that normalization solves alpha1's stability problem.

Across 10 seeds the raw alpha1 control ranges from 0.78 to 0.88 Q.ANT accuracy. Reference and Q.ANT predictions remain closely aligned: the raw control averages only 0.3 disagreements per 100 test cases. The numerical backend transition therefore remains a secondary limitation on this benchmark.

The next experiment should stop spending the small ECG200 test set on incremental preprocessing variants. The larger gap is task performance: alpha1 remains materially below the Exp001 MLP baseline while retaining its compact 3,890-parameter design.

## Conclusion

**Result classification: supported_with_qualification.**

Supported: per-case z-score normalization satisfies the frozen Exp002 stabilization criterion.

Qualification: the gain in stability is small and accompanied by a small loss in mean accuracy. It does not establish z-score normalization as the preferred PNN-v1 preprocessing pipeline.

## Recommended next hypothesis

PNN-v1 Proposal003 should test one architecture-level change motivated by the current model structure, not a broad search. A residual linear bypass around the Q.ANT Fourier/KAN head is a useful next test: it preserves the Q.ANT-native nonlinear hierarchy while giving the classifier a simple low-complexity path for information that need not be represented through periodic nonlinearities.

To reduce test-set-guided iteration, candidate selection should be based on a deterministic validation split made only from the canonical TRAIN set. The canonical TEST set should be evaluated once after the choice is frozen.

No result here measures real photonic hardware latency, throughput, energy efficiency or optical noise.
