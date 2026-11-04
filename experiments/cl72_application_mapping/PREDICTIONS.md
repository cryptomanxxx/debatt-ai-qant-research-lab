# CL-72 predictions — which applications could benefit from Q.ANT's linear primitive (written before any computation)

Committed and pushed before any computation. Informal predictions, not a formal preregistration. This is a follow-up to the [research goal](../../docs/RESEARCH_ROADMAP.md#research-goal-set-by-the-owner-2026-11-03).

## Question

CL-68 to CL-71 showed that models built on `linear_fprop` keep their float32 accuracy on the Q.ANT CPU backend. That is a **necessary** condition for photonic hardware to be useful, not a demonstration of advantage: any speed or energy gain can only be measured on hardware (phase E). **Which application workloads best satisfy the necessary conditions for a photonic advantage, and which do not?**

## Necessary conditions (derived; the photonic speed and energy numbers are unknown)

| # | Condition | Measure per workload |
|---|---|---|
| C1 | Most compute is matrix–vector multiplication the photonic unit could do | Share of MACs in matrix products |
| C2 | The matrices are fixed weights, not activation × activation products that change with every input (like attention QKᵀ and AV), since reloading the optical weights may be costly (unknown) | Share of MACs with static weights |
| C3 | The layers are large enough to use a photonic matrix and to amortise data transfer | MAC-weighted mean fan-in; arithmetic intensity, i.e. MACs per bf16 byte of layer input and output with the weights resident |
| C4 | bfloat16-level numerics suffice | Measured in Part B for a CNN; already shown for MLPs (CL-69, CL-71) |
| C5 | Little work outside matrix products, such as normalisation, softmax, GELU and embedding lookups, which stays on the host | Count of non-matrix elementwise operations per MAC |

## Part A — analytic workload accounting (a calculation, not an experiment)

Standard published shapes, one inference:

- an MLP classifier (784–256–256–10);
- LeNet-style CNN (as in Part B);
- ResNet-18 at 224²;
- MobileNetV2 at 224²;
- BERT-base encoder at sequence lengths 128 and 2,048;
- GPT-2-small decode, one token at context 1,024;
- a DLRM-style recommender (26 embedding tables of dimension 64, bottom MLP 13–512–256–64, top MLP 415–512–256–1);
- a 2-layer LSTM (hidden size 512, input size 512).

The counts come from layer formulas in `run.py`; nothing is trained.

## Part B — measured fidelity of a CNN on the simulated Q.ANT path (Tier 0)

- **Model.** A LeNet-style CNN on Fashion-MNIST: conv 5×5×16, ReLU, avg-pool, conv 5×5×32, ReLU, avg-pool, then linear 512→10. Convolutions are written as im2col matrix products, so every MAC goes through `linear_fprop`.
- **Training.** Adam lr 1e-3, batch 128, 3 epochs, seeds 11/22/33.
- **Q.ANT path.** It uses the CL-70-confirmed model of `linear_fprop` (bf16 operands and output, float32 accumulation), with bf16 bias and ReLU. Pooling is done on the host in float32 and recast to bf16.

## Predictions

1. **Best candidates by C1–C3:** ResNet-18 and the BERT encoder at sequence length 128, each with ≥ 95% of MACs in static-weight matrix products and a MAC-weighted mean fan-in ≥ 500.
2. **Worst candidates:**
   - MobileNetV2, because its depthwise convolutions have fan-in 9: its MAC-weighted mean fan-in is below 300.
   - DLRM, because per inference its embedding traffic in bytes exceeds its MACs.
3. **Long context hurts transformers:** for BERT at sequence length 2,048 and for GPT-2 decode at context 1,024, dynamic matrix products are ≥ 15% of MACs.
4. **The CNN keeps its float32 accuracy on the Q.ANT path:** a gap within ±0.1 pp and ≤ 20 disagreements per 10k.
