# CL-74 predictions — phase E pilot kit and CPU baselines (written before any timing run)

Committed and pushed before any timing run. Informal predictions, not a formal preregistration.

## Why

A photonic advantage can only show up as speed, latency or energy. The Q.ANT CPU backend cannot measure any of these, so further CPU-backend runs are paused ([roadmap](../../docs/RESEARCH_ROADMAP.md)). This step prepares for phase E. It has two parts:

1. **A pilot kit.** The same benchmark runs on any backend that provides a matrix product: NumPy float32 here, and Q.ANT's toolkit on the pilot hardware. It ships with fixed trained weights and reference outputs, so the pilot needs no training and no dataset download.
2. **CPU baselines** for the same workloads, measured now. This gives the photonic numbers something to be compared against.

## Workloads

| Name | Model | Trained on | Input |
|---|---|---|---|
| `mlp` | 784–256–256–10, ReLU (CL-69 `mlp_h256x2`) | Fashion-MNIST, 3 epochs, seed 11 | 28×28 image |
| `cnn` | LeNet-style im2col CNN (CL-72 Part B) | Fashion-MNIST, 3 epochs, seed 11 | 28×28 image |
| `encoder` | 2-layer BERT-style search encoder (CL-73) | repo corpus, 40 epochs, seed 11 | 128 tokens (BERT-128 length) |

## Measurements per workload, backend and batch size (1, 8, 64, 256)

- Latency per batch call: median and 95th percentile, after warm-up.
- Throughput, in inferences per second.
- Effective GMAC/s, from analytic MAC counts.
- The share of wall time spent inside matrix products.
- Weight-load time: preparing all weights in the backend's format. On the pilot this is the cost of reconfiguring the photonic weights.
- Fidelity against the stored float32 and simulated Q.ANT reference outputs: top-1 agreement and the maximum relative output error.
- Energy: `NOT_MEASURED`. The container has no power counter and no instrument. On the pilot, the kit computes joules per inference only from a measured power value supplied with `--watts`.

The CPU is a 4-core Intel Xeon at 2.30 GHz (cloud VM), with NumPy and OpenBLAS at 1 and 4 threads. There is no GPU in this environment, so the GPU baseline is `NOT_MEASURED` and is left as a separate step.

## Predictions (CPU, float32)

1. **Batching:** effective GMAC/s at batch 256 is at least 5× that at batch 1, for every workload with 1 thread.
2. **Where the time goes:** at batch 64, the matrix-product share of time is at least 70% for `mlp` and `encoder`, but below 60% for `cnn` (im2col and pooling run on the host).
3. **Latency:** `encoder` at batch 1 with 1 thread takes at most 10 ms per query. That is about 17M MACs.
4. **Threads:** at batch 256, 4 threads are between 1.5× and 4× faster than 1 thread for every workload.
5. **Fidelity:** the simulated Q.ANT path agrees with float32 on top-1 for at least 99% of the stored reference inputs of every workload. This follows from CL-72 and CL-73, so it is a check of the kit and not a finding.
