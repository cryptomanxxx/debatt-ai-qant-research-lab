# CL-9M — Static runner-policy preflight (NO MODEL / NO_GO)

CL-9M adds a **source-only**, fail-closed check of the already existing CL-9I no-model CI workflow against a frozen expected template and the CL-9L `NO_GO` profile. It is wired into the existing Continual Learning CPU Contracts job and performs no new model execution, network probe or training. The checker has no automatic way to authorize any future H1 runner.

## What this actually verifies

- The reviewed `.github/workflows/cl9i-static-h1.yml` text has not changed: it declares `contents: read`, `persist-credentials: false`, three-minute timeout, Python 3.11, two exact static test commands and the known events/actions.
- CL-9L machine-readable flags still deny training, inference and execution, and acknowledge no verified network isolation.
- Regression mutations to permission, checkout credentials, timeout, action identity, triggers, commands and NO_GO flags are rejected.
- The existing `continual-learning-contracts.yml` CI executes the source checker and mutation tests, not the model.

This is an **exact source allowlist**, not a YAML parser, transitive code review, effective permission check, GitHub administration settings read, network security audit, cost guarantee, or a general validator for user-supplied arbitrary workflows. It deliberately fails on any CL-9I workflow source change (including harmless changes) until reviewed and explicitly re-baselined.

## Explicit remaining limits

- Default GitHub-hosted runner internet egress **is not blocked or proven isolated**.
- Historical workflows with secrets and write permissions remain outside this narrow test.
- Tag-pinned Actions in the existing static test workflow are not full SHA-pinned; immutable supply-chain requirements for a future H1 runner remain open.
- No approved H1 work metric, numerical tolerance, dataset digests, artifact writer, independent scientific signoff or exact Marcus authorization is provided.
- This PR does not amend the `verified_no_network_host` gate or implement a runnable H1 workflow.

**Decision: NO_GO. Source checks passing is not scientific readiness or permission to run models.**
