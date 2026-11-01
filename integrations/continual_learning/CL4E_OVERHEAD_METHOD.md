# CL-4E — future overhead measurement method (NO TRAINING)

The CL-4D Python `__getitem__` weight-read wrapper adds instrumentation work. Future wall-clock comparisons must distinguish this cost from the underlying dense/sparse forward paths, and cannot subtract an invented constant.

## Candidate paired method — **not yet executed**

- On identical fixed inputs and fresh fixed-weight model states, alternate **plain → instrumented → instrumented → plain** within short blocks (ABBA); the plain runs must execute the same forward method with no wrappers.
- Preregister warm-up, timer choice (monotonic high-resolution clock), process isolation, repeated blocks, placement of garbage collection, CPU contention checks and order effects before collecting data.
- Report both original plain timings, both instrumented timings, the within-block difference and ratio. Negative differences are possible due to noise and must not be silently clamped.
- Separate dense and sparse arms, activity strata and standalone inference from any later learning steps. Do not use the timer to infer CPU instructions, DRAM bytes or joules. No production GitHub Actions timing claim may be made from one uncontrolled runner.

## What is implemented now

`cl4e_overhead_contract.py` validates and summarizes **explicitly synthetic** ABBA timing rows in integer nanoseconds. Unit tests confirm deterministic arithmetic, sequence integrity, rejection of invalid metadata and signed differences. They never start a clock or a model. The numerical example is illustrative, not a result from the hardware, GitHub runners, Q.ANT or Intel Loihi.

## Scientific boundary

All nine CL-3Y gates remain open. The CL-3X H1 study, provenance, measurement-overhead calibration, quality parity, independent review and separate Marcus experiment authorization remain **NO_GO**. This PR does not authorize or measure real instrumentation overhead.

Run only metadata/unit tests: `python -m unittest -v tests.test_cl4e_overhead_contract`.
