# CL-8G — Execution host isolation decision (NO MODEL / NO TRAINING)

**Decision: NO_GO for every model workload.** This document records the outcome of two manual fixed-child Linux isolation smoke tests. It does not grant execution authority.

## Direct evidence

| GitHub Actions run | Observed result | Scope |
| --- | --- | --- |
| [41343095078](https://github.com/cryptomanxxx/debatt-ai-qant-research-lab/actions/runs/41343095078) | Failed closed; generic isolation-unavailable reason | Fixed no-model Linux namespace probe only |
| [41343841231](https://github.com/cryptomanxxx/debatt-ai-qant-research-lab/actions/runs/41343841231) | Failed closed; `OS_PERMISSION_DENIED` from `unshare` | Same probe, on GitHub-hosted `ubuntu-24.04` |

All probe unit tests passed in the observed runs. That demonstrates software contract behavior, **not** the security of a future model workload.

The evidence shows that `unshare --user --map-root-user --net` cannot establish the required network namespace under the permissions of the tested GitHub-hosted runner. It does **not** establish that all GitHub-hosted isolation techniques fail, nor does it imply an exploit or defect in GitHub Actions. We will not bypass the host restriction or run an unsandboxed model.

## Candidate architectures to review

1. **GitHub-hosted with externally enforced isolation.** Evaluate a supported runner/container architecture with a documented, externally enforced outbound deny. Before relying on it, use a *fixed non-model* test that attempts outbound connection to a controlled endpoint and verifies denial. Network namespace inode inequality is insufficient. Do not use privileged mode or assume a default container is offline.
2. **Controlled research server / self-hosted runner.** Configure OS-level egress filtering and execution budgets at the host boundary; independently audit the runner, filesystem and artifact isolation. A self-hosted runner is not automatically safe. Do not expose privileged host administration to PR code or untrusted inputs.

Neither alternative is implemented or approved. GitHub Actions remains suitable for existing static reports, deterministic CI, and no-model contract tests.

## Required before an actual experiment

- An independently reviewed host threat model, code pin, input digest and exact execution command.
- Reproducible, external egress-denial evidence from a *fixed non-AI* probe, plus CPU, memory, wall, disk/output, child-tree cleanup and immutable artifacts.
- A separately reviewed scientific protocol addressing learnability, pairing, leakage, quality and forgetting.
- Marcus's **separate explicit approval** for any proposed training/inference run with resource budget and reviewed commit.

The machine-readable [decision record](cl8g_host_isolation_decision.json) stays `NO_GO`. The CL-8C [future execution envelope](cl8c_cpu_execution_envelope.json) remains unimplemented and `NO_GO`.

**No model, inference, training, new data generation, network-bypass attempt or energy measurement is performed by CL-8G.**
