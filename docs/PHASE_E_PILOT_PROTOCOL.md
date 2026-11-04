# Phase E — measurement protocol for the Q.ANT photonic pilot

**Status:** prepared 2026-11-04 (CL-74). Running it is **Tier 2** and needs the owner's explicit approval of the exact scope.
**Kit:** [`experiments/cl74_pilot_kit`](../experiments/cl74_pilot_kit/README.md). It is standalone: NumPy and Q.ANT's toolkit only, with fixed weights and reference outputs included.

## Question

Is any of the three workloads faster, lower-latency or more energy-efficient on Q.ANT's photonic hardware than on a CPU (and GPU) at equal output fidelity?

## Workloads

These are the CL-72 candidates at a size the kit ships:

- **`encoder`**: a BERT-style search encoder, 128 tokens.
- **`cnn`**: a LeNet-style im2col CNN.
- **`mlp`**: 784–256–256–10.

## Steps on the pilot

1. **Fidelity first.** Run `python kit.py check --backend qant_toolkit`. Continue only if top-1 agreement with the stored float32 reference is ≥ 99% for every workload. Record the maximum relative error and the minimum cosine.
2. **Timing.** Run `python kit.py bench --backend qant_toolkit --out pilot.json`, and the same command with `--backend float32` on the pilot's host CPU. Batches 1, 8, 64 and 256.
3. **Energy, only if it can be measured:**
   - Measure the average power during step 2 with an instrument, or with Q.ANT's own power telemetry, and record which.
   - Rerun with `--watts <measured value>`.
   - Otherwise, energy stays `NOT_MEASURED`.
4. **Record:**
   - the hardware and toolkit versions;
   - which operations ran photonically and which ran on the host;
   - whether the host↔chip transfer is inside the timed call.

## What counts as an advantage

- **Measure end to end.** Use the wall time of the whole workload, including host operations, transfers and the weight load. Matrix-product time alone does not count.
- **Compare like with like.** Use the same workload, batch size and fidelity threshold as the CPU baseline from CL-74.
- **Use a fair baseline.** The CL-74 NumPy baseline is a lower bar. A claim also needs an optimised CPU runtime, such as ONNX Runtime, and a GPU baseline.
- **Report weight-load time separately.** If it exceeds about 100 inference calls, only workloads with fixed weights can benefit. That matches the CL-72 condition C2.
- **Report all results.** A slower or less efficient result is a valid result and is reported the same way.

## Questions to settle with Q.ANT before the run

1. Which toolkit operations run photonically on the pilot (`linear_fprop`, `add_bias_fprop`, activations, softmax)?
2. What is the physical matrix size, and how are larger matrices tiled?
3. What do a weight reconfiguration and a host↔chip transfer cost?
4. Can the device extract convolution patches itself, or must im2col patches be sent?
5. Can power or energy be measured or logged during the pilot?
