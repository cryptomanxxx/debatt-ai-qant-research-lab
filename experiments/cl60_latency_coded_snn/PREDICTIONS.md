# CL-60 predictions — latency-coded spiking network on MNIST (written before any run)

Committed and pushed before any run. Informal predictions, not a formal preregistration.

## Question

In CL-59, rate coding made input spikes the main operation cost: T × 103.9 × 256 additions per image. That was more than the zero-skipping dense network (41,247 MACs) at every T. **With latency coding, where each nonzero pixel spikes exactly once, does a spiking network reach dense accuracy with fewer synaptic operations than the zero-skipping network?**

## Setting

Everything is as in CL-59 (the same LIF network, BPTT and surrogate, Adam, 2 epochs, seeds 11/22/33, T = 4/8/16, dense ReLU reference) except for the encoding:

- A pixel with intensity x > 0 spikes once, at step t = round((1 − x)(T − 1)): bright pixels spike early.
- A pixel with x = 0 never spikes.
- The encoding is deterministic, so training and test inputs are the same on every pass.

## Derived before running (analytic, not a finding)

- **Input layer.** Input spikes per test image = nonzero pixels = 151.12 at every T, so the input layer costs 151.12 × 256 = **38,687 additions**. Rate coding at T=4 cost 106,394.
- **Step distribution.** Most spikes arrive at step 0: 77.4 of the 151.1 at T=4, and 61.2 at T=16.
- **Upper bound.** Hidden spikes are at most 256 × T per image, so total operations are at most 38,687 + 2,560 T. That gives 48,927 at T=4, 59,167 at T=8 and 79,647 at T=16. Every T is therefore below CL-59's cheapest rate-coded network (110,201) and below the dense network (203,264). This follows from the encoding, not from the run.
- **Break-even with zero-skipping.** To beat the zero-skipping network's 41,247 MACs, the network must emit at most (41,247 − 38,687) / 10 = **256 hidden spikes per image**. That is a hidden spike rate of at most 1/T: 0.25 at T=4, 0.125 at T=8 and 0.0625 at T=16.

## Predictions

1. Measured input spikes per image equal 151.12 (exact, up to float rounding).
2. Latency-coded SNN test accuracy is between 95.0% and 97.0% at every T, and at most 0.5 pp above the CL-59 rate-coded network at the same T.
3. Hidden spike rate is 0.10–0.30, below CL-59's 0.37–0.38.
4. **Main bet:** no T is within 1 pp of dense accuracy **and** below 41,247 total synaptic operations. In other words, the hidden layer spikes more than 256 times per image.
5. Training wall time per epoch is within ±30% of CL-59 at the same T, because NumPy dense matrix products do not exploit spike sparsity.
