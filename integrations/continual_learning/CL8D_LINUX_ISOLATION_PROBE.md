# CL-8D — Linux child isolation probe (NO MODEL / NO TRAINING)

**Status: PROTOTYPE; experiment remains NO_GO.** This adds a fixed, non-AI subprocess probe to check whether the execution environment permits a separate Linux network namespace and configured child resource ceilings. It is **not** a training runner, and it cannot accept a user-defined command, file, dataset, imported model, script or shell.

## Mechanism

- The parent launches only a fixed Python `-I -S -c` probe through Linux `unshare --user --map-root-user --net`.
- The child receives `RLIMIT_CPU=5 seconds`, `RLIMIT_AS=256 MiB`, `RLIMIT_FSIZE=8 KiB` and disabled core dumps. These limits apply **only to the disposable probe child**, not the GitHub runner or the later proposed experiment.
- An external 5-second wall deadline stops the child's process group on timeout.
- Success requires a valid fixed JSON response and a **different** `/proc/self/ns/net` network namespace inode from the parent.
- Unsupported user namespaces, missing `unshare`, non-Linux hosts, failed processes and unverifiable responses are **NO_GO**. The harness never falls back to a networked subprocess.
- The fixed probe neither opens network sockets nor provides proof that a *future general workload* will be safe.

## How to inspect (no model execution)

```bash
python -m unittest -v tests.test_cl8d_linux_isolation_probe
python -m integrations.continual_learning.cl8d_linux_isolation_probe --probe
```

The second command attempts only the fixed non-model probe. It returns status 2 if network namespaces are unavailable, which is a safe and expected outcome on some managed environments. It does **not** train a model.

## Still missing for a future H1 experiment

1. Independent threat review of the execution environment and hardening of any generic workload runner; this probe accepts no such workloads.
2. Reliable external output/disk budgets and immutable artifact writing (CL-8C's 20 MiB proposal remains **NOT IMPLEMENTED**).
3. A reviewed method for actually proving network isolation on the intended future host, including blocked outbound connectivity, not only a changed namespace identifier.
4. Separate scientific approval of data provenance, learnability calibration, numeric pairing, timing instrumentation and failure policy.
5. Marcus's **separate explicit authorization** for model inference/training at a specific reviewed commit, input digest and compute budget.

The existing CL-8C [execution envelope](cl8c_cpu_execution_envelope.json) correctly remains `NO_GO` and records that the *general* model sandbox is not implemented. CL-8D does not modify this decision or CL-6A's nine open gates.

**No models, training, inference, new data or physical-energy measurements are used here.**
