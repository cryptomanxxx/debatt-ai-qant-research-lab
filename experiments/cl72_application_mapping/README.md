# CL-72 — Which applications could benefit from Q.ANT's linear primitive?

**Tier 0** under [docs/RISK_TIERS.md](../../docs/RISK_TIERS.md), reusing the CL-58 Fashion-MNIST download; on a fresh checkout that download comes first, which is Tier 1. It runs locally in about 80 seconds: Part A is a calculation, and Part B uses the simulator only. It follows up the [research goal](../../docs/RESEARCH_ROADMAP.md#research-goal-set-by-the-owner-2026-11-03). [Predictions](PREDICTIONS.md) were pushed before any computation (commit `b41bd82`). Physical energy: `NOT_MEASURED`.

Reproduce from the repository root: `python -m experiments.cl72_application_mapping.run`, which writes [`results.json`](results.json).

## Question

CL-68 to CL-71 showed that `linear_fprop` models keep their float32 accuracy on the Q.ANT CPU backend. That is a **necessary** condition for a photonic advantage, not evidence of one, because speed and energy can only be measured on hardware (phase E). **Which application workloads best satisfy the necessary conditions, and which do not?**

## Method

**Part A (analytic).** One inference of each workload, with standard published shapes, is broken down into its matrix products. The layer formulas are in `run.py`; the tests check ResNet-18 (about 1.8 GMACs and 11.7 M parameters) and MobileNetV2 (about 0.30 GMACs) against commonly cited totals. Per workload:

- the share of MACs with **static weights** (C1, C2), as opposed to activation × activation products such as attention QKᵀ/AV or recommender interactions;
- the **MAC-weighted mean fan-in** and the minimum fan-in (C3);
- **arithmetic intensity**: MACs per bf16 byte of layer input and output, with the weights resident (C3). Embedding and KV-cache reads are counted as extra bytes.
- the bf16 weight size;
- **host operations per MAC**: activations, norms, softmax and residual adds (C5).

**Part B (simulated).** A LeNet-style CNN on Fashion-MNIST, with convolutions as im2col matrix products (3 seeds, 3 epochs). It is evaluated along the Q.ANT path modelled on CL-70, with bf16 operands and output and float32 accumulation in `linear_fprop`.

## Results

**Part A**

| Workload | GMACs | Static-weight share | MAC-weighted fan-in (min) | MACs per byte | Weights (MB, bf16) | Host ops per MAC |
|---|---|---|---|---|---|---|
| ResNet-18, 224² | 1.81 | **100%** | **1,828** (64) | **194** | 23 | 0.002 |
| BERT-base, sequence 128 | 11.2 | **97.3%** | **1,497** (64) | **197** | 170 | 0.003 |
| 2-layer LSTM, hidden 512 (per step) | 0.004 | 100% | 1,024 (1,024) | 341 | 8 | 0.002 |
| MLP 784–256–256–10 | 0.0003 | 100% | 650 (256) | 74 | 0.5 | 0.002 |
| DLRM-style recommender | 0.0005 | 95.7% | 438 (13) | 36 | 1 | 0.003 |
| LeNet-style CNN (Part B) | 0.001 | 100% | 319 (25) | 36 | 0.03 | 0.013 |
| MobileNetV2, 224² | 0.30 | 100% | **251 (9)** | **11** | 7 | 0.021 |
| GPT-2-small decode, context 1,024 | 0.14 | **86.7%** | 1,196 (64) | **3.7** | 247 | 0.007 |
| BERT-base, sequence 2,048 | 251 | **69.2%** | 1,388 (64) | 79 | 170 | 0.014 |

**Part B: CNN on the simulated Q.ANT path.** Float32 accuracy is 0.8427 and Q.ANT-path accuracy 0.8428, a gap of +0.01 pp, with 8.3 disagreements per 10k (10 / 9 / 6 per seed).

## Predictions

| # | Prediction | Outcome |
|---|---|---|
| 1 | ResNet-18 and BERT at sequence 128 have ≥ 95% static MACs and fan-in ≥ 500 | ✓ 100% / 1,828 and 97.3% / 1,497 |
| 2 | MobileNetV2 fan-in < 300; DLRM embedding bytes exceed its MACs | ✗ MobileNetV2 ✓ (251, with depthwise fan-in 9), but DLRM ✗ (3.3 kB of embeddings against 0.5 M MACs per inference) |
| 3 | BERT at sequence 2,048 and GPT-2 decode both have ≥ 15% dynamic MACs | ✗ BERT ✓ (30.8%), but GPT-2 decode only 13.3% |
| 4 | The CNN keeps its float32 accuracy on the Q.ANT path (gap ±0.1 pp, ≤ 20 disagreements) | ✓ +0.01 pp, 8.3 |

**2 of 4 held.** The DLRM miss: its embedding *traffic per inference* is small. Real recommenders are limited by the *size* of the embedding tables (many GB), which is a memory-capacity problem this per-inference count does not capture. The GPT-2 miss: at context 1,024 the attention products are 13% of MACs, but its arithmetic intensity (3.7) is the lowest of all workloads, so the conclusion that it suits photonics poorly still holds, for a different reason.

## Findings

1. **The strongest candidates are dense, static-weight vision and encoder workloads.** ResNet-type CNNs and short-sequence transformer encoders such as BERT at sequence 128:
   - put 97–100% of their MACs in fixed-weight matrix products;
   - have large fan-in (about 1,500–1,800) and high arithmetic intensity (about 195 MACs per byte);
   - need little host-side work.

   A CNN written as matrix products also keeps its float32 accuracy on the Q.ANT path (Part B).
2. **LSTMs and large MLP layers also meet C1–C3,** with the highest arithmetic intensity (341). But an LSTM must compute its time steps one after another, so its benefit depends on photonic *latency*, which is unknown.
3. **Weak candidates, each failing a different condition:**
   - **MobileNetV2-style networks:** depthwise convolutions have fan-in 9 and an arithmetic intensity of 11. Efficient-on-CPU designs are a poor fit.
   - **Long-context transformers:** at sequence 2,048, 31% of BERT's MACs are activation × activation products whose "weights" change with every input.
   - **Autoregressive decoding (GPT-2):** about 4 MACs per byte. Each 247 MB set of weights is used once per token, so the workload is data-movement-bound unless the weights stay on the chip.
   - **Recommenders:** the MLP part is small, and the main cost, the embedding tables, is memory, not matrix work.
4. **Ranking as test candidates for the photonic pilot (phase E):**
   1. CNN image classification (ResNet-type);
   2. short-sequence transformer encoders;
   3. LSTM or time-series models, if latency is good;
   4. long-context or decode transformers, and depthwise CNNs, last.

## What this does and does not show

It ranks workloads by **necessary** conditions only.
- Whether any of them is faster or more energy-efficient on Q.ANT's photonic hardware depends on facts we do not have: the physical matrix size, the cost of reconfiguring weights, host↔chip transfer, analog noise and power.
- Part A rests on standard shapes and simple accounting, without measurements. Batch-size effects on weight reuse are not modelled; the intensity counts activation I/O with resident weights.
- Part B uses the CPU-backend model of `linear_fprop` from CL-70, not the photonic chip.
