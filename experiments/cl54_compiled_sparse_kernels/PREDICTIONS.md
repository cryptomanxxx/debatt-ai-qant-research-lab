# CL-54 predictions (written before any timing run)

Committed and pushed before the timing experiment was run. Correctness checks on random weights were run during development, but no timing comparisons. These are informal predictions, not a formal preregistration.

## Background

In CL-53 (NumPy), the low-rank kWTA inference path did 0.79× the counted work of dense H=256 but took 40× longer, because the gather-and-einsum path cannot use an optimised dense matrix multiply. **Question:** with compiled kernels, does counted work predict wall time?

## Setting

- **Models:** retrain the CL-53 models for seed 11 with the CL-53 code (MNIST, 2 epochs): dense H=128 and H=256, and kWTA H=1024 with k=32 or 64 using exact or low-rank selection.
- **Kernels:** written in C, compiled with `gcc -O3 -march=native -ffast-math`, single-threaded, one sample at a time (matrix–vector), float32. The same flags apply to every kernel.
- **Kernel variants:**
  - `dense`: full W1·x.
  - `dense_sparse_input`: skips zero pixels (about 19% are nonzero), using a column-major copy of W1.
  - `kwta_exact`: all 1024 units, then top-k.
  - `kwta_lowrank`: Vᵀx, then A·z, then top-c candidates computed exactly, then top-k.
  - `kwta_lowrank_sparse_input`: the same, also skipping zero pixels.
- **Reference:** the NumPy BLAS batched path from CL-53.
- **Timing:** inference over the 10k test set, median of 7 repeats, with an A/A control.
- **Correctness:** predictions agree with the NumPy reference on at least 99.9% of test samples.

## Predictions

1. C `dense` per-sample H=256 is slower than NumPy's batched BLAS H=256, by 1.5–5×.
2. **Main bet:** for the compiled kernels, wall time relative to C `dense` H=256 tracks counted work relative to dense H=256 within a factor of 2 for every kWTA variant. This would contrast with the 40–87× gap in NumPy.
3. C `kwta_lowrank` k=32 is faster than C `dense` H=256 (time ratio < 1).
4. `dense_sparse_input` takes 0.2–0.5× the time of C `dense` at the same H, and gives the same predictions on ≥ 99.9% of samples.
5. The fastest kWTA variant (`kwta_lowrank_sparse_input`, k=32) is faster than NumPy BLAS dense H=256 per test set.
6. The A/A difference is under 10%.
