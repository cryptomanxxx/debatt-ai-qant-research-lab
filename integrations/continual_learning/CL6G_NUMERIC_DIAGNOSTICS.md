# CL-6G — Numerical difference diagnostics (NO MODEL EXECUTION)

**Status: NO_GO. No model inference, initialization, training, synthetic-data generation or hardware measurement.**

CL-6F describes what a separately approved future stepwise model-state comparison must retain. CL-6G implements a **pure function** to describe differences between *caller-supplied scalar sequences*. It does not load model weights, import model implementations, evaluate examples or perform any experiment.

## Metrics (descriptive, not acceptance criteria)

For each pair of finite scalars `a`, `b`:

- Absolute difference: `abs(a - b)`.
- Symmetric relative difference: `abs(a - b) / max(abs(a), abs(b))`, with exact zero pairs defined as 0.
- Aggregate fields: number of compared scalars, exact-equality count, maximum absolute and maximum symmetric-relative difference.

The symmetric relative metric is **scale-sensitive near zero** and must not be read as a general statistical uncertainty measure. It also does not quantify scientific impact on labels, loss or training trajectories. A future reviewed protocol may need additional summary statistics and per-tensor diagnostics.

## Fail-closed conditions

Inputs must be equally long, nonempty Python lists or tuples of finite real-valued scalars. Mismatched lengths, booleans, NaN, infinities, lossy integer-to-float conversions (including integers above the binary64 exact-integer range), unusable integer magnitudes and overflowing differences are rejected; they are not silently dropped. Nested tensors and data loading are **not** part of this helper.

**No numerical tolerance or equivalence pass/fail threshold is approved.** The helper always reports `scientific_decision=NO_GO` and `equivalence_established=false`. Using it with literal unit-test values is not a model run or a substitute for a future paired scientific experiment.

Before any real model-state use, an independent review must approve the capture boundary, frozen inputs, per-field comparison scope, tolerances and uncertainty. Marcus must separately authorize the relevant execution, especially model training. All nine CL-6A readiness gates remain open.

See [CL-6F stepwise protocol](CL6F_STEPWISE_STATE_DIFF_PROTOCOL.md).
