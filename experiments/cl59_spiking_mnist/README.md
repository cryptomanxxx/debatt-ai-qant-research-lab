# CL-59 — A spiking (LIF) network on MNIST

**Tier 1** under [docs/RISK_TIERS.md](../../docs/RISK_TIERS.md). It reuses the CL-53 MNIST download (pinned SHA-256). Single-threaded NumPy, 3 processes, about 80 seconds in total. **Exploratory.** [Predictions and analytic operation counts](PREDICTIONS.md) were pushed before the run (commit `f839eac`); a 3,200-sample pilot for speed was run after that. Physical energy: `NOT_MEASURED`. This is a CPU simulation, not neuromorphic hardware.

Reproduce from the repository root: `python -m experiments.cl59_spiking_mnist.run`, which writes [`results.json`](results.json).

## Question

This is the first temporal spiking network in the repository. **Does a rate-coded LIF network reach dense-network accuracy on MNIST, and how many synaptic operations does it use compared with a dense network and with the zero-skipping network from CL-57?**

## Setup

- **SNN:** 784 → 256 LIF neurons → 10 non-spiking readout averaged over T = 4, 8 or 16 steps.
  - Leak β = 0.9, threshold 1, reset by subtraction.
  - Input uses Bernoulli rate coding: each pixel spikes with probability equal to its intensity.
  - Training uses backpropagation through time with a fast-sigmoid surrogate gradient.
- **Dense reference:** 784 → 256 ReLU → 10.
- **Training (both):** Adam (lr 1e-3, batch 64), 2 epochs, seeds 11, 22 and 33.
- **Checks:** BPTT gradients match finite differences when the spike function is replaced by a smooth sigmoid (`tests/test_cl59_spiking_mnist.py`).
- **Counts per test image:**
  - SNN synaptic operations (additions) = input spikes × 256 + hidden spikes × 10.
  - Dense MACs = 203,264.
  - Zero-skipping dense MACs = 41,247 (151.1 nonzero pixels).

## Results

Mean over 3 seeds (per-seed values are in `results.json`).

| Model | Test accuracy | Operations per image | Hidden spike rate | Epoch time |
|---|---|---|---|---|
| Dense ReLU, H=256 | 97.08% (97.04 / 96.87 / 97.32) | 203,264 MACs; 41,247 with zero-skipping | — | 2.0 s |
| SNN, T=4 | 96.65% (96.72 / 96.69 / 96.53) | 110,201 additions | 0.372 | 5.1 s |
| SNN, T=8 | 96.77% (96.75 / 96.75 / 96.81) | 220,607 additions | 0.380 | 8.5 s |
| SNN, T=16 | 96.80% (96.85 / 96.74 / 96.80) | 441,218 additions | 0.382 | 16.2 s |

Measured input-layer operations match the analytic T × 103.9 × 256 within 0.01% (for example 106,393 against 106,385 at T=4).

## Predictions

| # | Prediction | Outcome |
|---|---|---|
| 1 | Input-layer operations within 2% of the analytic count | ✓ within 0.01% |
| 2 | Dense 97–98%; SNN ≥ 2 pp below at T=4, within 1.5 pp at T=16 | ✗ dense ✓ (97.08%) and T=16 ✓ (−0.28 pp), but T=4 is only 0.43 pp below |
| 3 | Hidden spike rate 5–25% | ✗ 37–38% |
| 4 | **No T is within 1 pp of dense accuracy and uses fewer operations than the zero-skipping network** | ✓ T=4 is within 1 pp but needs 110k additions against 41k MACs |
| 5 | SNN operations below dense MACs only at T=4 | ✓ 110k < 203k; T=8 is already above (221k) |
| 6 | Training time ≥ T× dense | ✗ 2.6× (T=4), 4.3× (T=8), 8.1× (T=16) |

**3 of 6 held.**

## Findings

1. **The spiking network learns almost as well as the dense one.** It reaches 96.65–96.80% against 97.08%, a gap of 0.3–0.4 pp at 2 epochs, already at T=4. More time steps barely help.
2. **Rate coding makes it do more operations, not fewer, than a zero-skipping dense network.** At T=4 it needs 110k additions against 41k multiply-accumulates. Every bright pixel spikes again at each step, so the input layer alone costs T × 103.9 × 256. This was predictable on paper; the experiment confirms it.
3. **Hidden activity is not sparse:** 37–38% of neuron–step pairs spike. Nothing in the training pushes towards sparse firing.
4. **Whether additions are cheap enough to make up for 2.7× more operations is a hardware question,** and was not tested. It depends on the energy of an addition against a multiply-accumulate and on event routing costs on a specific chip such as Loihi. On this CPU the SNN is simply slower.

## Next steps that could change the picture

- **Fewer input spikes:** latency coding (at most one spike per pixel) or delta coding would cut the input-layer term, which dominates the count.
- **Sparse firing:** a penalty on the hidden spike rate during training, measured as an accuracy–operations trade-off.
- **Hardware:** only a neuromorphic or event-driven platform can test energy (Tier 2).
