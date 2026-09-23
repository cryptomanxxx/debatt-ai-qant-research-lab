# Experiment 004 — Depth & Topology Search

This experiment follows Experiment 003 by keeping ReLU and varying network topology.

Candidates:
- 784 → 64 → 10
- 784 → 64 → 64 → 10
- 784 → 128 → 10
- 784 → 128 → 64 → 10
- 784 → 256 → 10
- 784 → 128 → 128 → 10

The controlled comparison records Q.ANT/reference accuracy, parameter count,
prediction disagreements and diagnostic CPU time. A Pareto front maximizes
Q.ANT accuracy while minimizing parameter count.

Run:

```bash
python experiments/004_depth_topology/run.py
```

Outputs are written to `local_results/`.
