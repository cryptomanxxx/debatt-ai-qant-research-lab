# Debatt-AI Photonic Neural Network v1

This directory is the development workspace for **Debatt-AI Photonic Neural Network v1 (PNN-v1)**.

PNN-v1 is a new research series built on the findings from the earlier Q.ANT Toolkit discovery experiments. The historical Toolkit experiments and their identifiers remain unchanged.

## Research objective

Develop an original neural-network architecture around Q.ANT Native Computing Toolkit nonlinear operations, then evaluate whether the resulting architecture can solve practical problems. Time-series analysis is the first planned application area.

Current development uses Q.ANT's software/simulation path. Results from this phase must not be interpreted as measurements of real photonic latency, throughput, energy use, or other hardware performance. Those require later validation on Q.ANT photonic hardware.

## Numbering

PNN-v1 starts a new, synchronized series:

- PNN-v1 Proposal001 → PNN-v1 Exp001 → Result001 → Analysis001
- PNN-v1 Proposal002 → PNN-v1 Exp002 → Result002 → Analysis002

Historical Exp001–Exp030 and proposal identifiers outside this directory are not renamed or moved.

## Structure

- `experiments/` — reproducible PNN-v1 experiment implementations
- `proposals/` — preregistered hypotheses and experiment proposals
- `results/` — structured experiment results
- `analyses/` — AI Researcher analyses
- `models/` — PNN-v1 model definitions and architecture code
- `benchmarks/` — baseline and comparative benchmark definitions/results
- `datasets/` — dataset manifests, loaders and documentation; large datasets should not be committed

This separation preserves the provenance of the Toolkit discovery phase while giving PNN-v1 a clean development history.
