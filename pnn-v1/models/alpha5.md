# Debatt-AI Photonic Neural Network — Alpha5

## Status

**Active PNN-v1 architecture**, promoted by preregistered PNN-v1 Exp020.

## Architecture

For the ECG200 reference configuration:

```text
96 inputs
  ↓
6 contiguous windows × 16 samples
  ↓
6 independent Q.ANT Fourier/KAN blocks: 16 → 16
  ↓
concatenate → 96 features
  ↓
3 independent Q.ANT Fourier/KAN readout heads: 96 → 2
  ↓
arithmetic mean of the three logit vectors
  ↓
2-class prediction
```

Frequency basis: `k=[1,2]`.

Parameter count on ECG200: **8,550**.

## Why Alpha5 exists

Alpha5 changes only Alpha4's final readout. The single Q.ANT-native head is replaced by three independently parameterized Q.ANT-native heads whose logits are averaged.

The intervention was motivated by Exp017–019, where redundant readout consistently reduced mean absolute reference/Q.ANT logit error across TwoLeadECG, GunPoint, and SonyAIBORobotSurface1 without a demonstrated accuracy penalty under the relevant preregistered tests.

Exp020 then tested the model revision directly against Alpha4 on ECG200.

## Promotion evidence

Exp020:

- Alpha4 mean Q.ANT accuracy: **89.30%**
- Alpha5 mean Q.ANT accuracy: **89.70%**
- Alpha4 logit MAE: **0.03264**
- Alpha5 logit MAE: **0.02001**
- Alpha4 aggregate Q.ANT correct: **893 / 1000**
- Alpha5 aggregate Q.ANT correct: **897 / 1000**
- Alpha5 parameter count: **8,550**
- observed Alpha5 reference/Q.ANT prediction disagreements: **0 / 1000**

All preregistered promotion gates passed.

## Scope

Alpha5 is an experimental software/simulation-backend architecture. These results do not establish latency, energy efficiency, throughput, noise robustness, or other performance on physical Q.ANT photonic hardware.
