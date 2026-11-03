# CL-53 predictions (written before any run)

These predictions were committed before the experiment was run. They are informal predictions, not a formal preregistration.

## Background

CL-52 showed that, with 16 sparse binary inputs, choosing the active hidden units cheaply costs as much as it saves, so a small dense network wins. The open question is whether that changes when each unit is expensive: MNIST has 784 inputs per unit.

## Setting

- **Data:** MNIST, the full 60k training and 10k test sets, downloaded at run time (Tier 1, owner-approved). SHA-256 digests are recorded in the results.
- **Model:** 784 → H → 10 with ReLU, minibatch SGD (batch 64, learning rate 0.05), 2 epochs, 3 seeds. Hyperparameters are fixed in advance, with no tuning.
- **Arms:**
  - dense networks with H = 32, 64, 128, 256 and 512 (the accuracy–work frontier);
  - kWTA with H=1024 and k = 32 or 64, using either exact selection (all 1024 units computed) or low-rank screening (rank 32, with 4k candidates computed exactly).
- **Work:** forward weight terms per inference sample, counted with a dense fan-in of 784.
- **Timing:** single-threaded NumPy inference wall time on the 10k test set, as the median of repeated runs, plus an A/A control (the same dense model timed twice).

## Predictions

1. Dense accuracy: H=32 about 95%, H=256 about 97.5%, H=512 about 97.8%.
2. Exact kWTA with H=1024, k=32: about 97.5%, at about 4× the work of dense H=256.
3. Low-rank kWTA with k=32: accuracy within 0.5 percentage points of exact kWTA, at no more than 0.25× its work, with top-k recall of at least 0.8.
4. **Main bet:** low-rank kWTA becomes competitive with the dense frontier, unlike in CL-52, but does **not** beat it by more than seed noise at equal work.
5. **Wall time:** low-rank kWTA inference is at least 1.5× slower than a dense network with similar counted work, because gathering scattered weight rows is less efficient than one dense matrix multiplication.
6. The A/A timing difference is under 10%.
