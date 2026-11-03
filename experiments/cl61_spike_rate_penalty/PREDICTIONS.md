# CL-61 predictions — spike-rate penalty for the latency-coded SNN (written before any run)

Committed and pushed before any run. Informal predictions, not a formal preregistration.

## Question

In CL-60, the latency-coded SNN at T=4 used 41,842 synaptic operations per image, 1.4% more than the zero-skipping dense network (41,247 MACs). The hidden layer emitted 316 spikes per image; the budget to beat zero-skipping is 256. **Does a spike-rate penalty bring the SNN below the zero-skipping network, at what accuracy cost, and does it beat smaller dense networks?**

## Setting

- **SNN:** the latency-coded network from CL-60 at T=4, with H=256. Everything else is unchanged.
- **Loss:** cross-entropy + λ · (hidden spike rate), where the rate is spikes / (H · T) per image, averaged over the batch.
  - λ ∈ {0, 0.03, 0.1, 0.3, 1, 3}.
  - The penalty gradient flows through the same surrogate.
- **Extra arm:** the latency SNN with H=64 and λ=0.
- **Baselines:** dense ReLU networks with H = 64, 128 and 256, counted with zero-skipping (CL-57).
- **Training:** 2 epochs, Adam, seeds 11/22/33, as in CL-59/60. Hyperparameters were fixed in advance and not tuned.

## Derived before running (analytic, not a finding)

- **Floor for the SNN.** The input layer costs nonzero pixels × H = 151.12 × H additions, whatever λ is. For H=256 that is 38,687, so no λ can take the H=256 SNN below 38,687 operations.
- **Zero-skipping dense networks** cost 151.12 × H + 10 H MACs: 10,312 for H=64, 20,623 for H=128 and 41,247 for H=256.
- **Consequence.** A dense H=64 network does about 3.8× fewer operations than the H=256 SNN's floor. The SNN can only win on this count if it is more accurate than the smaller dense networks.

## Predictions

1. The hidden spike rate falls monotonically with λ.
2. At least one λ > 0 gets total operations below 41,247 (at most 256 hidden spikes per image) with test accuracy ≥ 95.0%.
3. No λ gives accuracy within 1 pp of the dense H=256 network (97.08% in CL-60).
4. **Main bet:** the dense H=64 network reaches ≥ 95.5% with 10,312 MACs. It is therefore more accurate than every penalized H=256 SNN with ≤ 256 hidden spikes, at about a quarter of the operations.
5. The latency SNN with H=64 is at least 1 pp less accurate than the dense H=64 network.
