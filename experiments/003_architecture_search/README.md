# Experiment 003 — Multi-dimensional architecture search

This experiment moves from a one-dimensional width sweep to a small search
space with two architectural variables:

- hidden width: 16, 32, 64, 128, 256
- activation: ReLU or Sigmoid

That produces 10 candidates under controlled training conditions.

The program records a fitness vector for every candidate and automatically
computes a Pareto front with two objectives: maximize Q.ANT accuracy and
minimize parameter count.

Results are written to `local_results/` rather than the tracked `results/`
directory. This prevents generated Colab files from blocking future
`git pull` operations. Selected verified results can later be committed to
`results/`.

Run from repository root:

```bash
python experiments/003_architecture_search/run.py
```

CPU timing is diagnostic only and is not treated as photonic hardware
performance or used in the initial Pareto calculation.
