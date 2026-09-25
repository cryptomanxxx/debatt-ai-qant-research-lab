# PNN-v1 Analysis004 — hierarchical cross-window mixer

## Status

PNN-v1 Exp004 completed successfully on the Q.ANT CPU/software-simulation backend. Architecture selection reused the deterministic TRAIN/validation boundary introduced in Exp003; canonical TEST remained confirmatory.

## Observed results

| Candidate | Parameters | Mean validation Q.ANT accuracy | Validation std |
|---|---:|---:|---:|
| alpha1 control | 3,890 | 0.685 | 0.05025 |
| alpha2 hierarchical mixer | 5,066 | 0.635 | 0.08958 |

The preregistered success criterion was not met. Alpha1 remained selected. Its confirmatory TEST mean Q.ANT accuracy was 0.802 with standard deviation 0.02676.

## Interpretation

Adding a second low-fan-in Q.ANT Fourier/KAN mixing stage did not recover useful cross-window structure. It reduced mean validation accuracy by 0.05 absolute and substantially increased seed sensitivity despite adding 1,176 parameters.

Together with Exp003, this rules out two immediate extensions of alpha1 under the current protocol: a linear residual classifier path and an additional neighboring-window nonlinear hierarchy.

Reference/Q.ANT agreement remains close. The more productive next question is therefore whether alpha1 compresses each eight-sample local window too aggressively when mapping 8 inputs to only 4 learned Fourier/KAN features.

## Conclusion

**Result classification: not_supported.**

The hierarchical-mixer hypothesis is falsified under the preregistered Exp004 protocol. Alpha1 remains the active architecture.

## Recommended next hypothesis

Change only the local feature width while keeping the single-stage alpha1 topology. Compare the current 8→4 local blocks with wider 8→6 and 8→8 blocks. This directly tests the information-bottleneck hypothesis without adding depth.

Candidate selection should continue to use the frozen TRAIN/validation boundary. Canonical TEST should be evaluated only for the candidate selected by a rule fixed before execution.

No result here measures real photonic hardware latency, throughput, energy efficiency or optical noise.
