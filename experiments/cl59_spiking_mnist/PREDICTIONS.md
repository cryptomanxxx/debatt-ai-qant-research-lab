# CL-59 predictions — spiking (LIF) network on MNIST (written before any run)

Committed and pushed before any run. Informal predictions, not a formal preregistration.

## Question

The README motivates event-driven, spiking computation, but no temporal spiking network has been tested in this repository. **On MNIST, does a spiking network reach dense-network accuracy, and with how many synaptic operations compared with a dense network and with the zero-skipping network from CL-57?**

## Setting

- **Data:** MNIST, reusing the CL-53 download (pinned SHA-256).
- **SNN:** 784 → 256 LIF neurons → 10 non-spiking readout.
  - Leak β = 0.9, threshold 1, reset by subtraction.
  - Input uses Bernoulli rate coding: each pixel spikes with probability equal to its intensity, at every time step.
  - Logits are the readout averaged over T steps, with T = 4, 8 or 16.
  - Training uses backpropagation through time with a fast-sigmoid surrogate gradient (slope 5).
- **Dense reference:** 784 → 256 ReLU → 10.
- **Training (both):** Adam (lr 1e-3, batch 64), 2 epochs, seeds 11, 22 and 33, single-threaded NumPy. Hyperparameters were fixed in advance and not tuned.
- **Operation counts per test image:**
  - SNN synaptic operations: input spikes × 256 plus hidden spikes × 10, summed over T. Neuron state updates (256 × T) are reported separately.
  - Dense network: multiply-accumulates (MACs), 784 × 256 + 256 × 10 = 203,264.
  - Zero-skipping dense network: 151.1 nonzero pixels × 256 + 2,560 ≈ 41,242 MACs.

## Derived before running (analytic, not a finding)

The mean pixel intensity sum per test image is 103.9, so the expected input spikes per step are 103.9. Input-layer synaptic operations are then T × 103.9 × 256:

| T | Expected input-layer synaptic operations |
|---|---|
| 4 | 106,394 |
| 8 | 212,787 |
| 16 | 425,574 |

That compares with 200,704 MACs for the dense first layer and 38,682 for the zero-skipping first layer. In other words, beyond T ≈ 1.5 rate-coded spiking does **more** input-layer operations than the zero-skipping network. Its operations are additions rather than multiply-adds, which is the usual argument for SNNs on neuromorphic hardware. That argument is not tested here.

## Predictions

1. Measured input-layer synaptic operations match the analytic values within 2%.
2. Dense test accuracy is 97.0–98.0%. SNN accuracy rises with T: at T=4 it is at least 2 pp below the dense network, and at T=16 it is within 1.5 pp.
3. Hidden neurons spike on 5–25% of (neuron, step) pairs, averaged over the test set.
4. **Main bet:** no T reaches dense accuracy within 1 pp **and** uses fewer total synaptic operations than the zero-skipping network's MACs.
5. SNN total synaptic operations are below the dense network's MACs only at T=4.
6. Training wall time per epoch is at least T× that of the dense network.
