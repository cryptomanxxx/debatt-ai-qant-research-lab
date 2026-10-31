# CL-3E: measurement scope and calibration prerequisites

**NO_GO / NO TRAINING.** This adds only an *estimated source-level
measurement ledger*, not an experimental runner or a wattmeter.

## What is estimable now

With a hypothetical fixed number of updates and active inputs per update,
existing CL-2 source-loop profiles can be multiplied into **projected**
counts of parameter reads, writes, forward weight terms, input checks and
assumption-dependent packed payload-byte equivalents. The ledger tags every
projection with `cl3e-source-loop-estimate-v0` and includes bias accesses
in parameter totals.

These are **not** empirical observations. The assumed constant activity
is not the per-example activity of the synthetic dataset. Payload proxies
do not represent DRAM traffic or physical memory-system energy.

## What remains unmeasured

- Training and held-out inference require **separate** activity and access
  ledgers. Inference accounting is not yet implemented; never reuse training
  profiles as inference measurements.
- Nonlinearity math, hidden-state memory, cache misses, Python object memory,
  index-allocation costs and CPU instruction/time overhead are not fully
  counted in the present source-level model.
- Actual CPU energy requires hardware-aware, calibrated measurements
  (idle subtraction, measurement interval, device/CPU scope, repeatability).
- GPU, Loihi, Q.ANT, photonic I/O, DAC/ADC and memory conversion costs cannot
  be inferred from these software counters.
- The existing CL-3A scientific-review, dataset and independently approved
  bounded-runner gates still stand. In particular this PR does **not**
  authorize training or close any blocker.

## Future review gates

A subsequent reviewed change must define a single consistent accounting
boundary for both model arms, add inference and nonlinear/index overhead,
quantify uncertainties and document the meter if physical energy is attempted.
A separate human authorization tied to a specific code/protocol version is
required before any experimental training.
