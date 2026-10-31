# Optional CL-1 simulation toolchains (development-only)

**Status:** availability integration, not a validated benchmark and not compute authorization. The original Toolkit/PNN-v1/Dev-4/v3 dependencies, workflows, immutable evidence and guarded approval path are unchanged.

This directory hosts a no-side-effect check for three independently installed tools:

| Tool | Location | Intended later use | Current check |
| --- | --- | --- | --- |
| Q.ANT Native Computing Toolkit | Existing `scripts/setup_qant_cpu.sh` and historical experiment scripts | CPU/software backend for photonic-native operators | Python module discoverable |
| Intel [Lava](https://github.com/lava-nc/lava) | New *separate* virtual environment | Spiking neuron/process and local-learning experiments | Python module discoverable |
| [ScopeX-ASU SimPhony](https://github.com/ScopeX-ASU/SimPhony) | New *separate* source checkout | Architecture-level electrical/photonic energy estimation | Checkout contains `onnarchsim/simulator.py` |

## Read-only readiness probe

From repository root:

```bash
python -m integrations.continual_learning.check_backends
python -m unittest discover -s tests -p test_continual_learning_backend_availability.py
```

Optionally pass `--simphony-root /path/to/ScopeX-SimPhony` (or set `DEBATT_AI_SIMPHONY_ROOT`); `--strict` gives a nonzero exit code when a tool is absent. No installation, training, API calls, benchmarking or hardware access is performed by this probe. **Availability is not a working integration validation.**

## Separate installation environments for later manual validation

Do not modify or upgrade the Q.ANT/PNN environments: its experiments have versioned, pinned provenance. For Lava, follow its [official installation guide](https://lava-nc.org/lava/notebooks/in_depth/tutorial01_installing_lava.html) in a fresh venv. Use the documented CPU backend: this is not access to an Intel Loihi chip.

For SimPhony, obtain the **ScopeX-ASU/SimPhony** GitHub source, not the unrelated BYU project named `simphony`. Follow [the upstream prerequisites](https://github.com/ScopeX-ASU/SimPhony#prerequisites) in a separate environment: its requirements, CACTI memory simulator and optional TorchONN/pyutility are additional dependencies. Merely cloning does not establish that SimPhony runs.

The Q.ANT module can be provisioned with the existing, historically used `scripts/setup_qant_cpu.sh` in its own appropriate runner. This document does not execute that script.

## Before CL-1 experiments

1. Confirm platform, Python version, licenses, dependency provenance, and upstream commit SHAs separately; record any install or smoke-test failures.
2. Add **small genuine CPU execution smoke tests** for each backend, with explicit numerical reference tests and no network on test invocation.
3. Verify SimPhony's energy model and configuration boundaries, calibrate laser/modulator/ADC/DAC/memory/control terms, and show uncertainty intervals.
4. Preregister dataset, sequence, baselines, repetitions and a quality-matched energy/retention comparison in an **independent CL-1 protocol**.
5. Obtain scientific review and separate human compute authorization before executing training.

Do not present CPU utilization, SimPhony estimates or Q.ANT CPU-backend behavior as measured physical-photonic energy or performance. See [the CL-1 draft](../../docs/continual-learning-architecture-comparison-draft.md) and [current governance map](../../docs/current-systems-and-protocols.md).
