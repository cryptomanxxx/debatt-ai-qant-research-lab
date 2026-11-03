# CL-61 — Spike-rate penalty for the latency-coded SNN

**Tier 1** under [docs/RISK_TIERS.md](../../docs/RISK_TIERS.md). It reuses the CL-53 MNIST download (pinned SHA-256). Single-threaded NumPy, 3 processes, about 80 seconds in total. **Exploratory.** [Predictions and analytic counts](PREDICTIONS.md) were pushed before the run (commit `449ad48`). Physical energy: `NOT_MEASURED`. This is a CPU simulation, not neuromorphic hardware.

Reproduce from the repository root: `python -m experiments.cl61_spike_rate_penalty.run`, which writes [`results.json`](results.json).

## Question

In [CL-60](../cl60_latency_coded_snn/README.md), the latency-coded SNN at T=4 needed 41,842 operations per image, 1.4% more than the zero-skipping dense network. **Does a spike-rate penalty bring it below that network, at what accuracy cost, and does it beat smaller dense networks?**

## Setup

- **SNN:** the latency-coded SNN from CL-60 at T=4, with H=256.
  - Loss = cross-entropy + λ · hidden spike rate, for λ ∈ {0, 0.03, 0.1, 0.3, 1, 3}.
  - The penalty gradient flows through the surrogate.
  - An extra arm uses H=64 with λ=0.
- **Baselines:** dense ReLU networks with H = 64, 128 and 256. Their cost is counted as zero-skipping MACs, 151.12 × H + 10 H.
- **Training:** 2 epochs, Adam and seeds 11/22/33, as in CL-59/60.
- **Checks:** `tests/test_cl61_spike_rate_penalty.py` compares the gradients, with the penalty included, against finite differences, using a smooth spike stand-in.
- **Derived before the run:** the H=256 SNN can never go below 38,687 operations, because each of the 151.12 nonzero pixels × 256 costs one addition.

## Results

Mean over 3 seeds (per-seed values are in `results.json`; seed spread ≤ 0.45 pp).

| Model | Test accuracy | Operations per image | Hidden spikes per image (rate) |
|---|---|---|---|
| Dense H=64, zero-skipping | 95.72% | 10,312 MACs | — |
| Dense H=128, zero-skipping | **96.61%** | **20,624 MACs** | — |
| Dense H=256, zero-skipping | 97.08% | 41,247 MACs | — |
| SNN H=256, λ=0 | 95.70% | 41,842 additions | 315.5 (0.308) |
| SNN H=256, λ=0.03 | 95.78% | 41,388 | 270.1 (0.264) |
| SNN H=256, λ=0.1 | 96.03% | 40,684 | 199.7 (0.195) |
| SNN H=256, λ=0.3 | 96.23% | 39,989 | 130.2 (0.127) |
| SNN H=256, λ=1 | **96.42%** | **39,355** | 66.8 (0.065) |
| SNN H=256, λ=3 | 96.00% | 39,036 | 34.9 (0.034) |
| SNN H=64, λ=0 | 93.69% | 10,822 | 115.1 (0.449) |

## Predictions

| # | Prediction | Outcome |
|---|---|---|
| 1 | Spike rate falls monotonically with λ | ✓ from 0.308 to 0.034 |
| 2 | Some λ > 0 gets below 41,247 operations at ≥ 95.0% | ✓ λ = 0.1–3, at 96.0–96.4% |
| 3 | No λ within 1 pp of dense H=256 | ✗ λ=1 is 0.66 pp below |
| 4 | **Dense H=64 (≥ 95.5%) beats every penalized SNN on accuracy** | ✗ H=64 reaches 95.72%, but the penalized SNNs reach 96.0–96.4% |
| 5 | SNN H=64 at least 1 pp below dense H=64 | ✓ 2.0 pp below |

**3 of 5 held.**

## Findings

1. **The penalty helps accuracy as well as sparsity.** From λ=0 to λ=1, hidden spikes fall 4.7× (316 to 67) and accuracy rises by 0.7 pp (95.70% to 96.42%), consistently across all three seeds. A plausible but untested reading is that the penalty acts as a regularizer in this short 2-epoch training. At λ=3 accuracy starts to fall again.
2. **The SNN gets below the zero-skipping H=256 network, but only slightly,** because the input layer sets a floor. At λ=1 it needs 39,355 additions against 41,247 MACs (−4.6%), at 0.66 pp lower accuracy. The input layer is 98% of that count.
3. **A smaller dense network still wins on operation count.** Dense H=128 with zero-skipping is more accurate (96.61% against 96.42%) at about half the operations (20,624 MACs against 39,355 additions). For the SNN to win on cost, an addition must cost less than about 0.52 of a MAC, together with its event-routing overhead. That is a hardware question and was not tested.
4. **A narrower SNN does not fix this.** With H=64 it loses 2.0 pp against the dense H=64 network at about the same operation count.

## What this does and does not show

On MNIST, with a 2-epoch budget and counted operations, a latency-coded, penalized LIF network is close to a dense network on accuracy and operations, but it is dominated by a dense network of half the width. Operation counts are not energy. The comparison between additions and MACs, and the routing cost of sparse events, can only be settled on real hardware (Tier 2). The H=64 SNN was not penalized, and longer training was not tried.
