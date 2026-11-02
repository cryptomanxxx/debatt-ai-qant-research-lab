# CL-10B — First H1 training core, gated and not executed

The new `cl10b_gated_h1_runner.py` contains an in-memory scientific core for a *future approved* 16–8–2 sequential-learning workload. It uses the existing four model classes and the shared-rule activity-shift generator, and prepares 5 checkpoints × 4 heldout tasks per seed/arm. It is **not a complete experiment runner**: the CLI always raises `ExecutionNotAuthorized`, no `workflow_dispatch` training workflow is added, and the only CI tests inspect static scope and demonstrate that the CLI refuses.

## Implementation boundary

- `_future_seed_arm_core(seed, arm)` is **private-by-convention**, a candidate for later independent review. It is not called from CI or the script entrypoint. Python does not provide a security sandbox against someone manually importing and calling it; the actual security boundary remains the absence of a runnable workflow, existing repository permissions and a future separately audited execution controller.
- The core would instantiate one arm for a seed, construct the four proposed datasets, perform at most 512 updates, evaluate all four heldout tasks at five checkpoints, and return in-memory accuracy/training records.
- The core does **not** provide the independently approved six-component source-event measurement, numerical parity traces, frozen dataset digests, train-only calibration, immutable artifacts, host network isolation or physical energy measurement. It deliberately returns `observed_h1_source_event_vector: null`.
- A standalone invocation reads the CL-10A review-only JSON, confirms `NO_GO`, prints only the static planned scope and fails before any model/dataset import.
- The existing `Continual Learning CPU Contracts` workflow executes only `test_cl10b_gated_h1_runner.py`, which never invokes the core. A green check is no experimental result.

## Scientific review before activation

1. Validate dataset construction feasibility across seeds, calibration split and train-only learnability; freeze per-seed digests in a separate authorized process.
2. Verify same-initialization and numerical equivalence of local dense/sparse/shared-index implementations, including approved tolerance and output/post-update traces.
3. Implement a fail-closed, size-bounded append-only artifact writer, full failure ledger and source-event instrumentation.
4. Independently resolve network boundary and exact runner resource limit, and pin source/environment/dependencies.
5. Obtain independent scientific/security signoff, then Marcus's **separate exact** approval for specific workload and budget, before implementing/enabling a training workflow.

CL-10B does not close any of these gates or claim H1/energy evidence. **NO_GO.**
