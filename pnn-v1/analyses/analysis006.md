# PNN-v1 Analysis006 — full-TRAIN Alpha2 confirmation

## Status

Exp006 completed on the Q.ANT CPU/software-simulation backend. Alpha1, frozen Alpha2, and MLP were trained on all 100 ECG200 TRAIN cases across the same 10 seeds.

## Results

| Model | Parameters | Mean accuracy | Std |
|---|---:|---:|---:|
| Alpha1 w4 (Q.ANT) | 3,890 | 0.824 | 0.03169 |
| Alpha2 w8 (Q.ANT) | 7,778 | 0.834 | 0.02154 |
| MLP-ReLU reference | 8,354 | 0.876 | 0.00800 |

Alpha2 had 0.835 mean reference accuracy and 0.1 mean reference/Q.ANT prediction disagreements per 100 test cases.

## Numerical boundary

The intended threshold was at least 0.01 absolute improvement. The displayed difference is exactly 0.010, but binary floating-point representations made the implementation comparison evaluate false by a tiny rounding difference. The recorded Exp006 result must remain unchanged: success_criteria_met=false and retain_alpha1. It should not be rewritten after seeing the data.

## Conclusion

**Result classification: boundary_case_requires_confirmation.**

A frozen confirmation using integer correct-prediction counts avoids binary floating-point ambiguity. The MLP remains stronger, so Alpha2 promotion would not imply parity with the conventional baseline.

No result here measures real photonic hardware latency, throughput, energy efficiency or optical noise.
