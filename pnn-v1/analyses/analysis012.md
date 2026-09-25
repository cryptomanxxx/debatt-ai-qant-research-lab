# PNN-v1 Analysis012 — Alpha4 cross-dataset transfer

## Result

Exp012 froze the Alpha4 design principles discovered on ECG200 and transferred them to GunPoint without target-dataset architecture search. GunPoint's 150 samples were deterministically center-cropped to 144, producing nine contiguous 16-sample windows. Each local Q.ANT Fourier/KAN block remained 16→16 with k=[1,2], followed by a 144→2 Q.ANT Fourier/KAN head.

Across ten preregistered seeds:

- Mean Q.ANT TEST accuracy: **94.93%**
- Q.ANT accuracy std: **1.24 percentage points**
- Aggregate Q.ANT correct: **1,424 / 1,500**
- Mean reference accuracy: **94.93%**
- Mean reference/Q.ANT prediction disagreements: **0.2 per 150 TEST cases**
- Mean absolute logit error: **0.03331**
- Alpha4-transfer parameters: **10,514**
- Fixed MLP calibration: **92.53%** mean TEST accuracy, 11,426 parameters

The preregistered transfer criterion (Q.ANT accuracy ≥0.90 and mean disagreements ≤1.0 per TEST set) was satisfied.

## Interpretation

**Classification: supported.**

This is stronger evidence than another ECG200 architecture search because the local receptive-field principle and compact k=[1,2] basis were frozen before the GunPoint TEST result. The result supports cross-dataset transfer of the tested Alpha4 design principles from ECG200 to GunPoint.

It does not establish broad time-series generality. Two datasets are insufficient for that claim. A multi-dataset benchmark is the appropriate next scientific step.

The MLP result is calibration only and was not part of the success gate.

All Q.ANT measurements use the CPU/software simulation backend. No physical photonic latency, throughput, energy, or hardware-performance claim is made.
