# CL-54 — Compiled sparse kernels: does counted work become wall time?

**Tier 1** under [docs/RISK_TIERS.md](../../docs/RISK_TIERS.md): it uses MNIST, downloaded at run time. Single-threaded CPU, a few minutes. **Exploratory.** [Predictions](PREDICTIONS.md) were committed and pushed (commit `2c3af94`, merged in #322) before any timing run. Physical energy: `NOT_MEASURED`.

Reproduce from the repository root: `python -m experiments.cl54_compiled_sparse_kernels.run`. This compiles [`kernels.c`](kernels.c) with `gcc -O3 -march=native -ffast-math` and writes [`results.json`](results.json) and [`timing.json`](timing.json).

## Question

In CL-53 (NumPy), the low-rank kWTA inference path did 0.79× the counted work of dense H=256 but took 40× longer. **With compiled kernels, does counted work predict wall time, and does any form of sparsity give a real speed-up?**

## Setup

- **Models:** the CL-53 models retrained for seed 11 with the CL-53 code. All six reproduce their CL-53 test accuracy exactly.
- **Kernels:** C, one sample at a time, float32, the same flags for all. Each model is run with the full input and with zero pixels skipped (on average 151.1 of 784 pixels are nonzero).
- **Agreement:** every kernel gives the same prediction as the NumPy reference on **100%** of the 10k test samples.
- **Timing:** full 10k test set, median of 7 repeats, on a shared cloud CPU (x86-64, gcc 13.3), with an A/A control.

## Results

Work and time are relative to C dense H=256 with full input (344 ms).

| Kernel | Accuracy | Counted work | Wall time | Time ÷ work |
|---|---|---|---|---|
| dense H=128 | 93.28% | 0.50 | 0.52 | 1.04 |
| dense H=128, sparse input | 93.28% | 0.10 | 0.17 | 1.71 |
| dense H=256 | 93.46% | 1.00 | 1.00 | 1.00 |
| **dense H=256, sparse input** | **93.46%** | **0.20** | **0.20** | **1.01** |
| kWTA exact, k=32 | 93.08% | 3.96 | 4.43 | 1.12 |
| kWTA exact, k=32, sparse input | 93.08% | 0.77 | 1.08 | 1.40 |
| kWTA exact, k=64, sparse input | 93.57% | 0.77 | 1.09 | 1.42 |
| kWTA low-rank, k=32 | 93.14% | 0.79 | 1.12 | 1.43 |
| kWTA low-rank, k=32, sparse input | 93.14% | 0.29 | 1.02 | 3.56 |
| kWTA low-rank, k=64 | 93.46% | 1.28 | 1.87 | 1.46 |
| kWTA low-rank, k=64, sparse input | 93.46% | 0.38 | 1.73 | 4.49 |
| *NumPy BLAS, dense H=256 (batched)* | 93.46% | 1.00 | *0.17* | — |
| *NumPy, kWTA low-rank, k=32 (gather)* | 93.14% | 0.79 | *6.43* | — |

**A/A control:** C dense H=256 was timed twice at 344 ms and 294 ms (15%); NumPy dense H=256 at 58 ms and 82 ms (41%). The shared CPU is noisy, so only differences of several-fold should be trusted.

## Predictions

| # | Prediction | Outcome |
|---|---|---|
| 1 | C dense H=256 is 1.5–5× slower than NumPy BLAS | ✗ 5.9× slower (344 ms against 58 ms) |
| 2 | Time tracks counted work within a factor of 2 for every kWTA kernel | ✗ 6 of 8 within 2×; low-rank with sparse input is 3.6–4.5× off |
| 3 | C low-rank k=32 is faster than C dense H=256 | ✗ 1.12× (and 1.02× with sparse input) |
| 4 | Skipping zero pixels takes 0.2–0.5× the time of C dense, with ≥ 99.9% agreement | ✓ 0.20× (H=256), 0.33× (H=128), 100% agreement |
| 5 | Fastest kWTA beats NumPy BLAS dense H=256 | ✗ 352 ms against 58 ms |
| 6 | A/A difference under 10% | ✗ 15% (C) and 41% (NumPy) |

## Findings

1. **With compiled kernels, counted work predicts time reasonably well.** For dense and kWTA kernels that read contiguous memory, time ÷ work is 1.0–1.5. The 40× gap in CL-53 came from NumPy's gather path, not from the method.
2. **Input sparsity gives the first real speed-up in the project.** Skipping zero pixels makes dense H=256 inference **5× faster** in C (70 ms against 344 ms), with identical predictions. This is H1's original idea, and it works when inputs really are sparse and the kernel exploits it.
3. **Optimised BLAS is still hard to beat.** NumPy's batched dense H=256 (58 ms) is about as fast as the sparse-input C kernel (70 ms), because BLAS handles many samples at once with cache blocking. A fair test of the sparse kernel would need a batched, tuned sparse implementation.
4. **Some kernels stay far slower than their counted work suggests.** Low-rank kWTA with sparse input is 3.6–4.5× off. Its candidate dot products read scattered positions within each weight row, and the H×r scoring plus two selections are not counted as expensive. Counting terms misses memory access patterns.
5. **kWTA still does not win.** The fastest kWTA kernel (352 ms, 93.14%) is 5× slower than sparse-input dense H=256 (70 ms, 93.46%).

## Implications for H1

- **Supported, narrowly:** with genuinely sparse inputs and a kernel that skips zeros, single-sample inference work and time fall in proportion (here 5×) at identical accuracy.
- **Not supported:** hidden-layer kWTA, with or without cheap selection, did not beat dense networks on time in any test from CL-50 to CL-54.
- **Before any energy work:** compare against a batched, tuned sparse library (e.g. sparse matrix–matrix products) rather than hand-written per-sample loops, on dedicated hardware with lower timing noise.
