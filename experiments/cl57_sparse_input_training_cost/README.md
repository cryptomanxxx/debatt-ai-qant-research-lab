# CL-57 — Training cost with sparse inputs

**Tier 1** under [docs/RISK_TIERS.md](../../docs/RISK_TIERS.md). It reuses the CL-53 MNIST download (pinned SHA-256). Single-threaded C and NumPy on CPU, a few minutes in total. **Exploratory.** [Predictions](PREDICTIONS.md) were committed and pushed before the timing run (commit `e7d9f37`). Physical energy: `NOT_MEASURED`.

Reproduce from the repository root: `python -m experiments.cl57_sparse_input_training_cost.run`. This compiles [`train_kernels.c`](train_kernels.c) and writes [`results.json`](results.json) and [`timing.json`](timing.json).

## Question

CL-54 showed that skipping zero pixels makes **inference** 5× faster, with one seed. In training, every weight attached to a zero input gets an exactly zero gradient. **Does the whole training step (forward, backward and weight update) get cheaper by the same factor, with an unchanged training trajectory?**

## Setup

- **Model and data:** MNIST, 784 → H → 10 with ReLU, H = 128 and 256. **3 seeds** (11, 22, 33) with the CL-53 initialisation.
- **Training:** 1 epoch of per-sample SGD at learning rate 0.01, with the same sample order for both kernels.
- **C kernels:** single-threaded, `gcc -O3 -march=native -ffast-math`. W1 is stored input-major.
  - `dense`: visits every input in the forward pass and updates every W1 column.
  - `sparse_input`: visits and updates only the nonzero pixels (149.9 of 784 on average).
- **Correctness:** a test checks both kernels against an independent NumPy implementation of the same SGD step, and checks that they are bit-identical to each other.
- **Reference:** NumPy minibatch SGD (batch 64, learning rate 0.05, dense BLAS). It is a different optimiser setting, so it is only a practical speed baseline.

## Results

Times are for one training epoch of 60k samples.

| H, seed | Weights bit-identical | Test accuracy | Dense | Sparse input | Time ratio | Counted work ratio |
|---|---|---|---|---|---|---|
| 128, 11 | ✓ | 96.66% | 1.76 s | 0.43 s | 0.24 | 0.21 |
| 128, 22 | ✓ | 96.20% | 2.20 s | 0.49 s | 0.23 | 0.21 |
| 128, 33 | ✓ | 96.53% | 1.34 s | 0.40 s | 0.30 | 0.21 |
| 256, 11 | ✓ | 96.61% | 3.73 s | 0.94 s | 0.25 | 0.21 |
| 256, 22 | ✓ | 96.69% | 3.22 s | 1.28 s | 0.40 | 0.21 |
| 256, 33 | ✓ | 96.71% | 2.97 s | 1.07 s | 0.36 | 0.21 |

- **NumPy batch-64 dense, H=256:** 0.99, 1.04 and 0.96 s per epoch, at 92.1–92.4% accuracy (fewer, larger updates).
- **Sparse-input C against NumPy:** 0.95–1.24× its time.
- **A/A (dense H=256, seed 11, run twice):** 3.73 s and 3.56 s, a 5% difference. Dense times differ by up to 25% between seeds on the shared CPU.

## Predictions

| # | Prediction | Outcome |
|---|---|---|
| 1 | Weights bit-identical between kernels in every seed | ✓ 6 of 6 |
| 2 | Sparse takes 0.2–0.4× the dense epoch time, in every seed | ✓ 0.23–0.40 |
| 3 | Time ÷ counted work within 1.5× between kernels | ✗ 4 of 6; 1.74 and 1.94 for H=256 seeds 22 and 33 |
| 4 | Accuracy ≥ 0.95 for H=256 | ✓ 96.6–96.7% |
| 5 | Sparse per-sample C is 0.5–2× NumPy batch-64 time | ✓ 0.95–1.24× |
| 6 | A/A difference < 20% | ✓ 5% |

## Findings

1. **Skipping zero inputs makes the whole training step 2.5–4.4× cheaper in time, with an identical result.** The weights after a full epoch are bit-for-bit the same as dense training in all 6 runs. This holds across 3 seeds and 2 sizes, which strengthens the single-seed inference result from CL-54.
2. **Counted work predicts most of the saving but not all of it.** Work falls to 0.21×; time falls to 0.23–0.40×. The extra cost comes from scanning for nonzero pixels and from the parts that do not shrink (output layer and biases), plus timing noise on the shared CPU.
3. **Per-sample sparse C training is about as fast as batched dense BLAS training in NumPy.** It needs the same time for 60,000 updates as NumPy needs for 938 batch updates, and it reaches higher accuracy after one epoch (96.7% against 92.3%), although the two optimiser settings are not directly comparable.

## Limitations

- One architecture (one hidden layer) and one dataset, with about 19% nonzero pixels. The benefit scales with input sparsity and would vanish for dense inputs.
- Deeper networks have hidden activations as inputs to later layers. These are often sparse after ReLU, but that was not tested.
- The sparse kernel was not compared with a tuned, batched sparse library.
- Timing was on a shared cloud CPU, single-threaded. This is not an energy measurement.

## Implication for H1

The one positive H1 result now covers **training as well as inference**: with genuinely sparse inputs, skipping zeros reduces training time by 2.5–4.4× across 3 seeds, at bit-identical results.
