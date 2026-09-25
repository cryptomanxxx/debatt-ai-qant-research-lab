# PNN-v1 Analysis011 — full-TRAIN confirmation of w16

## Result

Exp011 compared the active Alpha3 geometry (12×8→8) with the frozen Exp010 candidate (6×16→16). Both used k=[1,2], the full canonical 100-case ECG200 TRAIN split, the canonical 100-case TEST split, 100 epochs, and the same ten preregistered seeds.

| Model | Parameters | Mean Q.ANT TEST accuracy | Std | Correct / 1,000 | Mean ref/Q.ANT disagreements |
|---|---:|---:|---:|---:|---:|
| Alpha3 | 3,938 | 0.852 | 0.01661 | 852 | 0.1 |
| w16 candidate | 7,010 | 0.893 | 0.01418 | 893 | 0.3 |

The w16 candidate produced 41 more correct Q.ANT classifications across 1,000 predictions. The preregistered threshold was +10, and its across-seed standard deviation was lower than Alpha3's.

Reference accuracy was 0.896 for w16 and 0.851 for Alpha3. Mean absolute reference/Q.ANT logit error was 0.03264 for w16 versus 0.02617 for Alpha3.

## Interpretation

The full-TRAIN confirmation reproduces and strengthens the Exp010 signal: in this ECG200 architecture family, increasing the local temporal receptive field from 8 to 16 samples while keeping the concatenated representation width at 96 and k=[1,2] materially improves classification accuracy.

The improvement is not a parameter-efficiency result: w16 uses 7,010 parameters versus Alpha3's 3,938. It is an accuracy/stability result.

## Decision

**Classification: supported.**

The preregistered promotion rule is satisfied. Promote the frozen 6×16→16, k=[1,2] architecture to **PNN-v1 Alpha4**.

Because ECG200 has now been used repeatedly throughout architecture development, further tuning on this dataset carries increasing benchmark-adaptation risk. The next research phase should test whether the discovered architecture principles transfer to a second time-series dataset rather than continuing unrestricted ECG200 optimization.

All measurements are from the Q.ANT CPU/software simulation backend. No physical photonic latency, throughput, energy, or hardware-performance claim is made.
