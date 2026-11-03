# CL-60 — A latency-coded spiking network on MNIST

**Tier 1** under [docs/RISK_TIERS.md](../../docs/RISK_TIERS.md). It reuses the CL-53 MNIST download (pinned SHA-256). Single-threaded NumPy, 3 processes, about 70 seconds in total. **Exploratory.** [Predictions and analytic counts](PREDICTIONS.md) were pushed before the run (commit `b04cab3`). Physical energy: `NOT_MEASURED`. This is a CPU simulation, not neuromorphic hardware.

Reproduce from the repository root: `python -m experiments.cl60_latency_coded_snn.run`, which writes [`results.json`](results.json).

## Question

In [CL-59](../cl59_spiking_mnist/README.md), rate coding made input spikes the main cost. **With latency coding, where each nonzero pixel spikes exactly once, does a spiking network reach dense accuracy with fewer synaptic operations than the zero-skipping dense network?**

## Setup

- **Same as CL-59:** the LIF network (784 → 256 → 10, β = 0.9, threshold 1, reset by subtraction), BPTT with a fast-sigmoid surrogate, Adam (lr 1e-3, batch 64), 2 epochs, seeds 11/22/33 and T = 4/8/16.
- **Only the encoding changes:**
  - A pixel with x > 0 spikes once, at step round((1 − x)(T − 1)).
  - Zero pixels never spike.
  - The encoding is deterministic.
- **Tests:** `tests/test_cl60_latency_coded_snn.py` checks the encoder. The BPTT gradients are already checked in CL-59.
- **Derived before the run:**
  - The input layer costs exactly 151.12 × 256 = 38,687 additions at every T.
  - To beat the zero-skipping network's 41,247 MACs, the hidden layer may emit at most 256 spikes per image.

## Results

Mean over 3 seeds (per-seed values are in `results.json`). The rate-coded rows are copied from CL-59.

| Model | Test accuracy | Operations per image | Hidden spikes per image (rate) | Epoch time |
|---|---|---|---|---|
| Dense ReLU, H=256 | 97.08% | 203,264 MACs; **41,247** with zero-skipping | — | 1.9 s |
| Latency SNN, T=4 | **95.70%** (95.66 / 95.63 / 95.80) | **41,842** additions | 315.5 (0.308) | 4.6 s |
| Latency SNN, T=8 | 94.49% (94.57 / 94.45 / 94.44) | 43,847 additions | 516.0 (0.252) | 7.9 s |
| Latency SNN, T=16 | 93.23% (93.41 / 93.30 / 92.98) | 47,069 additions | 838.2 (0.205) | 13.9 s |
| Rate SNN, T=4 (CL-59) | 96.65% | 110,201 additions | 381 (0.372) | 5.1 s |

Measured input spikes per image are 151.12 at every T, exactly the nonzero-pixel count.

## Predictions

| # | Prediction | Outcome |
|---|---|---|
| 1 | Input spikes equal 151.12 per image | ✓ exact |
| 2 | Accuracy 95.0–97.0% at every T, and not more than 0.5 pp above rate coding | ✗ T=4 is 95.70%, but T=8 and T=16 fall to 94.5% and 93.2% |
| 3 | Hidden spike rate 0.10–0.30, below CL-59 | ✗ T=4 is 0.308; T=8 and T=16 are within range |
| 4 | **No T within 1 pp of dense and below 41,247 operations** | ✓ the best is −1.38 pp, at 41,842 operations (1.4% above) |
| 5 | Training time within ±30% of CL-59 at the same T | ✓ 10–14% faster |

**3 of 5 held.**

## Findings

1. **Latency coding cuts the operation count by 2.6×** compared with rate coding at T=4 (41.8k against 110.2k). That brings it to the same level as the zero-skipping dense network: 1.4% more operations, but additions instead of multiply-accumulates.
2. **It costs accuracy.** The best result, at T=4, is 1.4 pp below dense and 1.0 pp below rate coding.
3. **More time steps make it worse,** the opposite of rate coding: from 95.7% at T=4 to 93.2% at T=16. A hypothesis, not tested here: with leak β = 0.9, the information in early spikes decays over the steps, while the readout averages over all of them.
4. **The hidden layer is now the margin.** It emits 316–838 spikes per image, above the 256 needed to beat zero-skipping. Without a hidden spike-rate penalty, the spiking network does not get below the zero-skipping dense network on this count.
5. Whether 41.8k additions are cheaper than 41.2k MACs depends on hardware and was not tested.

## Next steps that could change the picture

- **Hidden spike-rate penalty** at T=4: the budget is about 256 hidden spikes per image.
- **Readout of the last step, or the membrane potential,** instead of the mean, together with a smaller leak. Either could test hypothesis 3 and recover accuracy at larger T.
