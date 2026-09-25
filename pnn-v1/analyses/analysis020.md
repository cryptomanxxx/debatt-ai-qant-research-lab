# PNN-v1 Analysis020 — Alpha5 promotion test on ECG200

## Result

Exp020 compared active Alpha4 against a candidate that changed only the final readout: three independently parameterized Q.ANT-native Fourier/KAN heads with averaged logits.

| Metric | Alpha4 | Alpha5 candidate |
|---|---:|---:|
| Parameters | 7,010 | 8,550 |
| Mean reference accuracy | 89.60% | 89.70% |
| Mean Q.ANT accuracy | 89.30% | 89.70% |
| Aggregate Q.ANT correct / 1000 | 893 | 897 |
| Q.ANT accuracy std | 1.42 pp | 2.45 pp |
| Mean reference/Q.ANT disagreements | 0.3 | 0.0 |
| Mean absolute logit error | 0.03264 | 0.02001 |

## Preregistered decision

**promote_alpha5**

All promotion gates passed:

1. Mean logit MAE was strictly lower.
2. Aggregate Q.ANT correct predictions did not fall by more than 10; they increased from 893 to 897.
3. Parameter count remained below 10,000.

The zero observed prediction disagreements are a secondary result and are not used to redefine the promotion decision. Accuracy standard deviation increased, so Alpha5 should not be described as uniformly better on every metric.

## Active architecture

Alpha5 is now the active PNN-v1 architecture:

**96 → 6 contiguous windows×16 → 6× Q.ANT Fourier/KAN 16→16 → concat96 → 3 independent Q.ANT Fourier/KAN 96→2 readouts → arithmetic mean of logits**

Frequency basis: **k=[1,2]**  
Parameter count on ECG200: **8,550**

## Interpretation

Exp017–019 established cross-dataset evidence for redundant readout as a numerical-agreement intervention. Exp020 converts that evidence into a model revision under a preregistered promotion gate.

The next research question should move beyond readout redundancy. A useful next target is the local representation itself: Alpha5 currently gives every local window the same fixed 16-dimensional output width. The next experiment can ask whether a narrower local representation preserves the promoted readout robustness while reducing model size, without changing the frequency basis or readout rule.

All measurements use the Q.ANT CPU/software simulation backend. No physical photonic-hardware performance claim is made.
