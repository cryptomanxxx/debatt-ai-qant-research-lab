# CL-53 — Hidden-layer sparsity and cheap winner selection on MNIST

**Tier 1** under [docs/RISK_TIERS.md](../../docs/RISK_TIERS.md): it downloads a public dataset (the owner approved this on 2026-11-03). Training is single-threaded NumPy on CPU and the whole run takes about 5 minutes. **Exploratory.** [Predictions](PREDICTIONS.md) were committed before the run (commit `6e08190`). Physical energy: `NOT_MEASURED`.

Reproduce from the repository root: `python -m experiments.cl53_mnist_hidden_sparsity.run`. If the data is missing, the script downloads MNIST into `data/`, which is ignored by git, and checks it against pinned SHA-256 digests. It writes [`results.json`](results.json) and [`timing.json`](timing.json).

## Question

CL-52 showed that with 16 sparse binary inputs, choosing hidden units cheaply saves nothing. **Does that change when each unit is expensive, as with MNIST's 784 inputs per unit?**

## Setup

- **Data:** MNIST, the full 60k training and 10k test sets.
- **Model:** 784 → H → 10 with ReLU, minibatch SGD (batch 64, learning rate 0.05), 2 epochs, 3 seeds (11, 22, 33). Hyperparameters were fixed before the run, with no tuning. A finite-difference gradient check confirms the backprop (relative error < 3×10⁻⁸).
- **kWTA:** keeps the k units with the largest pre-activations, then applies ReLU.
  - Exact selection computes all 1024 units.
  - Low-rank selection scores units through a rank-32 approximation of W1 and computes 4k candidates exactly. The approximation is refreshed every 50 batches.
- **Work:** forward weight terms plus selection operations per inference sample, using a dense fan-in of 784.
- **Timing:** single-threaded wall time for inference on the 10k test set, as the median of 7 runs. Low-rank inference uses a genuinely sparse path that gathers the candidate weight rows. An A/A control times dense H=256 twice.

A one-seed, one-epoch pilot with three arms ran after the predictions were committed; nothing was changed afterwards.

## Results

Test accuracy is the mean over 3 seeds (with sd). Work and time are relative to dense H=256 (203,264 terms; 55.5 ms).

| Arm | Accuracy % | Top-k recall | Work | Wall time |
|---|---|---|---|---|
| dense H=32 | 92.79 (0.25) | — | 0.12 | 0.23 |
| dense H=64 | 93.34 (0.48) | — | 0.25 | 0.47 |
| dense H=128 | 93.53 (0.31) | — | 0.50 | 0.63 |
| dense H=256 | 93.82 (0.37) | — | 1.00 | 1.00 |
| dense H=512 | 94.05 (0.26) | — | 2.00 | 2.51 |
| kWTA exact, H=1024, k=32 | 93.05 (0.23) | 1.00 | 3.96 | 4.8 |
| kWTA exact, H=1024, k=64 | 93.72 (0.21) | 1.00 | 3.96 | 6.0 |
| kWTA low-rank, H=1024, k=32 | 92.96 (0.17) | 0.98 | 0.79 | **40** |
| kWTA low-rank, H=1024, k=64 | 93.68 (0.34) | 0.99 | 1.28 | **87** |

**A/A control:** dense H=256 timed twice gave 55.5 ms and 49.0 ms, a 12% difference.

For low-rank k=32, about 59k of the 160k terms per inference are selection: 25k for projecting the input, 33k for scoring the units and about 1k for partitioning. Training adds about 41.5k terms per sample for the periodic refresh.

## Predictions

| # | Prediction | Outcome |
|---|---|---|
| 1 | Dense ≈ 95% (H=32), 97.5% (H=256), 97.8% (H=512) | ✗ Accuracy is 92.8–94.1%; 2 epochs at lr 0.05 is under-trained. Relative comparisons still hold, but absolute levels are low. |
| 2 | Exact kWTA k=32 ≈ 97.5%, at about 4× the work | ✗ accuracy (93.05%); ✓ work (3.96×) |
| 3 | Low-rank within 0.5 pp of exact, ≤ 0.25× its work, recall ≥ 0.8 | ✓ 0.09 pp, 0.20×, recall 0.98 |
| 4 | Low-rank competitive with the dense frontier but not better | ✓ not better; ✗ not competitive either: below the frontier |
| 5 | Low-rank inference ≥ 1.5× slower than dense at similar counted work | ✓ 40× slower (0.79× the counted work of dense H=256) |
| 6 | A/A difference < 10% | ✗ 12% |

## Findings

1. **kWTA networks sit below the dense frontier.** Low-rank k=32 (92.96% at 0.79× work) does no better than dense H=32 (92.79% at 0.12×). Low-rank k=64 (93.68% at 1.28×) is below dense H=256 (93.82% at 1.00×). In this regime, a smaller dense network is still the better use of work.
2. **Cheap selection itself works well.** Low-rank screening finds 98–99% of the true winners, loses only 0.04–0.09 pp against exact kWTA, and cuts counted work to 0.20–0.32× of exact. The selector is not what is failing: kWTA itself buys less accuracy per unit of work than a dense network does.
3. **Counted work and real time diverge sharply.** Low-rank inference does 0.79× the counted work of dense H=256 but takes 40× longer. Gathering scattered weight rows (`W1[cand]`, then `einsum`) cannot use an optimised dense matrix multiply. This shows **this NumPy implementation**, not a limit of the method: a compiled or GPU-specific sparse kernel would close part of the gap. It confirms the warning that operation counts do not predict time or energy.
4. **Timing noise is about 12%** in the A/A control. Only differences much larger than that are meaningful, which holds for the 4–87× effects here.

## Limitations

- Training is short (2 epochs, about 94% for dense). The conclusions could change with longer training or other hyperparameters; that was not tested.
- There are only 3 seeds. The kWTA–dense gaps are small (around 0.1–0.9 pp) compared with an sd of about 0.3 pp.
- The work counts assume a dense fan-in of 784. They do not use the roughly 19% pixel sparsity of MNIST.
- Timing is for NumPy on one shared cloud CPU core. It is not a hardware measurement.

## Implications for H1

- Across CL-50 to CL-53, no form of sparsity tested here has beaten a smaller dense network on accuracy per unit of work. Low-rank selection does solve the selection problem: it finds 98% of the right units at a fifth of the cost.
- The remaining open questions are (a) whether kWTA gains more with longer training or deeper networks, and (b) whether a compiled sparse kernel can make the counted savings real in time. (b) needs a different kind of tool (C or Numba) and is a natural CL-54 step before any energy measurement.
