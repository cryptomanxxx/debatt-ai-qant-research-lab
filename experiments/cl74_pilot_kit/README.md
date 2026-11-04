# CL-74 — Phase E pilot kit and CPU baselines

**Tier 0** under [docs/RISK_TIERS.md](../../docs/RISK_TIERS.md): local, about 1 minute for the baselines. `prepare.py` trains for about 8 minutes and reuses the CL-58 Fashion-MNIST download, which is Tier 1 on a fresh checkout. [Predictions](PREDICTIONS.md) were pushed before any timing (commit `392f2c1`). Physical energy: `NOT_MEASURED`. GPU: `NOT_MEASURED` (no GPU in this environment).

The owner paused further Q.ANT CPU-backend experiments, because a photonic advantage can only show up as speed, latency or energy ([roadmap decision](../../docs/RESEARCH_ROADMAP.md)). This step prepares the hardware pilot instead. The measurement protocol is in [docs/PHASE_E_PILOT_PROTOCOL.md](../../docs/PHASE_E_PILOT_PROTOCOL.md).

## What the kit is

[`kit.py`](kit.py) is standalone: it needs NumPy, plus Q.ANT's toolkit for the Q.ANT backend. It runs three workloads through any backend that provides a matrix product:

- `mlp`: 784–256–256–10;
- `cnn`: LeNet-style, with im2col;
- `encoder`: the CL-73 BERT-style search encoder at 128 tokens.

The backends are:

- `float32`: NumPy;
- `qant_sim`: the CL-70 model of `linear_fprop`;
- `qant_toolkit`: Q.ANT's `linear_fprop` and `add_bias_fprop`. It is written but **not executed** here, because running it is Tier 2.

[`weights.npz`](weights.npz) holds the trained weights (seed 11). [`reference.npz`](reference.npz) holds the inputs and the float32 and simulated Q.ANT outputs. The pilot needs no training and no dataset download.

| Command | What it does |
|---|---|
| `python kit.py check --backend B` | Top-1 agreement and output error against the stored references |
| `python kit.py bench --backend B --threads N [--watts W]` | Latency (median, p95 and first call), throughput, GMAC/s, matrix-time share and host weight-preparation time at batch 1, 8, 64 and 256. Energy is computed only from a measured `--watts`. |

Reproduce from the repository root:
1. `python -m experiments.cl74_pilot_kit.prepare` (or `--references-only`)
2. `python -m experiments.cl74_pilot_kit.run`, which writes [`results.json`](results.json).

The tests check that the kit's forward passes reproduce the CL-69, CL-72 and CL-73 models on both paths.

## Results

**Fidelity (the kit checked against itself).**
- `float32` reproduces the stored references exactly: error 0, agreement 100%.
- `qant_sim` agrees on top-1 with float32 for 100%, 99.9% and 100% of the reference inputs (`mlp`, `cnn`, `encoder`). Its maximum relative output error is 0.5%, and the encoder's minimum cosine is 0.99999.

**CPU baseline:** Intel Xeon @ 2.30 GHz (4-core cloud VM), NumPy 2.5 with OpenBLAS, float32.

| Workload | MACs per inference | Batch 1 latency (1 thread) | Throughput, batch 256 (1 / 4 threads), inferences/s | GMAC/s, batch 1 → 256 (1 thread) | Matrix-time share (batch 64) |
|---|---|---|---|---|---|
| `mlp` | 0.27 M | 0.021 ms | 245,700 / 584,200 | 12.9 → 66.1 | 94% |
| `cnn` | 1.05 M | 0.114 ms | 7,700 / 8,400 | 9.2 → 8.1 | 22% |
| `encoder`, 128 tokens | 16.8 M | 1.22 ms | 550 / 535 | 13.8 → 9.2 | 31% |

**Run-to-run noise.** On this shared VM, repeated runs differ by about 10%. The table is the final run after the Codex fixes. The run before it gave `mlp` 4.99× instead of 5.14× in prediction 1, and the encoder 1.08 ms.

**Weight loading.** `host_weight_prep_ms` (at most 0.003 ms) only measures converting weights on the host. The kit cannot see when the device loads weights: with `qant_toolkit` that happens inside the timed `linear_fprop` calls. So `device_weight_load_ms` is `NOT_MEASURED`. `first_call_ms`, the first call after preparation, is compared with the median and gives only a hint. Measuring the reconfiguration cost needs a toolkit operation or a statement from Q.ANT (protocol question 3). This was corrected after review: the first version claimed the column measured reconfiguration.

**Correction during the run.** In the first timing, the encoder spent 73% of its time in GELU, because NumPy's `x**3` is about 70× slower than `x*x*x`. The baseline would have been 4× too slow, which would have flattered any photonic result. The kit now uses `x*x*x`, and the reference outputs were rebuilt from the same weights. The numbers above are after the fix. The first run gave the encoder 4.6 ms at batch 1 and 198 inferences/s at batch 256. The `mlp` and `cnn` numbers did not change materially.

## Predictions

| # | Prediction | Outcome |
|---|---|---|
| 1 | GMAC/s at batch 256 ≥ 5× batch 1 for every workload (1 thread) | ✗ `mlp` 5.1× (4.99× in the previous run), `cnn` 0.88×, `encoder` 0.67× |
| 2 | Matrix-time share ≥ 70% for `mlp` and `encoder`, < 60% for `cnn` (batch 64) | ✗ `mlp` 94% ✓, `cnn` 22% ✓, `encoder` 31% ✗ |
| 3 | `encoder` batch 1 ≤ 10 ms (1 thread) | ✓ 1.22 ms |
| 4 | 4 threads 1.5–4× faster than 1 at batch 256, every workload | ✗ `mlp` 2.4× ✓, `cnn` 1.09×, `encoder` 0.97× |
| 5 | `qant_sim` top-1 agreement ≥ 99% on every workload | ✓ 100%, 99.9%, 100% |

**2 of 5 held.** The misses share one cause: in this NumPy implementation, the `cnn` and the `encoder` are limited by host-side work, not by matrix products. For the CNN that is im2col and pooling; for the encoder it is softmax, GELU and LayerNorm. Larger batches and more threads therefore do not help them.

## What this does and does not show

- **The kit is ready for the pilot.** It reproduces the float32 and simulated Q.ANT references and records everything the protocol asks for. The `qant_toolkit` backend has not been run. Its first real execution will be on the CPU backend or the pilot, both Tier 2, and its argument layout follows CL-71.
- **This baseline is a lower bar, not the fair one.** These are NumPy numbers on a shared cloud VM. An optimised CPU runtime with fused kernels, such as ONNX Runtime or PyTorch, and a GPU would both be faster. A photonic advantage claim needs those baselines too.
- **Host work matters for photonics as well.** Even with free matrix products, the NumPy encoder would only get 1/(1 − 0.31) ≈ 1.4× faster, and the CNN 1.3×. This is Amdahl's law: the speed-up is capped by the time spent outside matrix products. So the fused host operations, and any operations the device runs itself, decide much of the end-to-end result.
- Energy: `NOT_MEASURED`.
