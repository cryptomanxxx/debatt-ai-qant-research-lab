# Experiment 002 — Hidden-width architecture search

This is the lab's first systematic architecture sweep.

Six otherwise identical MNIST networks are trained with hidden widths:

`16, 32, 64, 128, 256, 512`

For every candidate the experiment records reference accuracy, Q.ANT
CPU-backend accuracy, prediction disagreements, parameter count and diagnostic
CPU execution time.

All candidates use the same dataset sizes, epoch count and random seed so the
width comparison is controlled.

## Run

From the repository root, after the Q.ANT CPU backend is installed:

```bash
python experiments/002_width_search/run.py
```

Outputs:

- `results/exp002_width_search.json`
- `results/exp002_width_search.csv`

The CSV is the first small fitness table for later Pareto analysis.

CPU timing must not be interpreted as Q.ANT photonic-hardware performance.
