# PNN-v1 Analysis007 — Alpha2 integer-count confirmation

## Status
Exp007 repeated the frozen full-TRAIN Alpha1/Alpha2 comparison from Exp006 using the preregistered integer correct-count gate.

## Results

| Model | Parameters | Correct / 1,000 | Mean Q.ANT accuracy | Std |
|---|---:|---:|---:|---:|
| Alpha1 w4 | 3,890 | 824 | 0.824 | 0.03169 |
| Alpha2 w8 | 7,778 | 834 | 0.834 | 0.02154 |

Alpha2 produced exactly 10 more correct Q.ANT classifications and had lower across-seed standard deviation. Mean reference/Q.ANT prediction disagreements were 0.1 per 100 cases for Alpha2.

## Decision
Both frozen promotion conditions were met. Exp007 records success_criteria_met=true and promotion_decision=promote_alpha2.

## Conclusion
**Result classification: supported.**

The frozen 8→8 architecture is promoted to active PNN-v1 Alpha2. This is evidence under the ECG200 protocol, not general superiority across datasets. Exp006's conventional MLP remained stronger at 0.876 mean accuracy versus Alpha2 0.834.

## Next direction
Treat Alpha2 as fixed baseline. Test whether the fixed frequency basis k=[1,2,3,4] is limiting task fit, using TRAIN-only validation for selection and canonical TEST only after selection.

No result here measures physical photonic latency, throughput, energy efficiency or optical noise.
