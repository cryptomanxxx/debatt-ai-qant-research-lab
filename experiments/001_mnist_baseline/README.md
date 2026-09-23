# Experiment 001 — MNIST baseline

Goal: establish the first reproducible baseline for the research lab.

The script trains a small two-layer classifier in PyTorch and evaluates the
same learned weights using Q.ANT Toolkit operations:

`Linear → ReLU → Linear → Softmax`

It records reference accuracy, Q.ANT CPU-backend accuracy, prediction
disagreements, parameter count, configuration and diagnostic CPU execution
time.

## Run after installing the Q.ANT CPU backend

From the repository root:

```bash
python -m pip install -r experiments/001_mnist_baseline/requirements.txt
python experiments/001_mnist_baseline/run.py
```

The result is written to `results/exp001_baseline.json`.

The default run intentionally uses 10,000 training images, 1,000 test images
and 3 epochs so the first experiment remains relatively small. Later
experiments can scale this up.

**Important:** CPU-backend timing is not evidence of Q.ANT photonic hardware
latency, throughput or energy efficiency.
