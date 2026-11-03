# CL-57 predictions — training cost with sparse inputs (written before any timing run)

These predictions were committed and pushed before the timing run. They are informal predictions, not a formal preregistration.

## Background

CL-54 showed that skipping zero pixels makes single-sample **inference** 5× faster, with identical predictions (one seed). In a training step, the gradient of every weight attached to a zero input is exactly zero. **Question:** does the backward pass and weight update also get cheaper, by about the same factor, with an unchanged training trajectory?

## Setting

- **Data:** MNIST, reusing the CL-53 download (pinned SHA-256).
- **Model:** 784 → H → 10 with ReLU, H = 128 and 256, the CL-53 He initialisation, **3 seeds** (11, 22, 33).
- **Training:** 1 epoch of per-sample SGD (batch 1), learning rate 0.01, the same sample order for both kernels.
- **C kernels:** single-threaded, using the CL-54 compiler flags. Both kernels store W1 input-major (W1ᵀ) and process inputs in the same order.
  - `dense`: every input contributes to the forward pass, and every W1 column is updated.
  - `sparse_input`: only nonzero inputs are visited in the forward pass and the W1 update.
- **Reference:** NumPy minibatch SGD (batch 64, dense BLAS). It is a different optimiser setting and is reported as a practical baseline only.
- **Timing:** one training epoch per seed and kernel, with an A/A control (the dense H=256 kernel run twice for seed 11).

## Predictions

1. **Identical training:** for each seed, the weights after one epoch are bit-for-bit identical between `dense` and `sparse_input`, so test accuracy is identical too.
2. **Training speed-up:** `sparse_input` takes 0.2–0.4× the epoch time of `dense` at the same H, in every seed.
3. **Counted work predicts time:** per-step time ÷ counted work stays within a factor of 1.5 of each other for the two kernels.
4. **Accuracy:** test accuracy after 1 epoch is ≥ 0.95 for H=256.
5. **Against BLAS:** `sparse_input` per-sample training (H=256) takes 0.5–2× the epoch time of NumPy batch-64 dense training.
6. **A/A:** the difference is under 20%.
