# CL-10F — CL-10E observed test evidence and static container boundary (NO_GO)

## Observed evidence (not a model experiment)

Manual GitHub Actions [run 41369061088](https://github.com/cryptomanxxx/debatt-ai-qant-research-lab/actions/runs/41369061088) completed successfully on **2026-11-02**, GitHub-hosted `ubuntu-24.04`, image version `20261025.353.1`, runner version `2.338.2`. The job `fixed-no-model-network-test` had one successful fixed probe step. It reported:

- `Host-controlled endpoint ready; checking positive bridge control`
- `Checking negative network-none control`
- `PASS: bridge control reachable, network-none container denied.`
- `SCOPE: container-only negative evidence; host is still internet-connected.`
- `DECISION: NO_GO; does not authorize training or inference.`

The BusyBox image was pulled by immutable manifest digest `sha256:fd7dc98638c8e305f4dc34e979f1c0fdfdcaeb0fbf8fcff77ae834b6da3d7e6e`. The positive control's success shows the fixed host-controlled HTTP endpoint was reachable under Docker bridge networking; the negative control's controlled failure under Docker `--network none` gives limited evidence of **container** network isolation. The test included no model, training, inference, dataset generation or physical-energy measurement.

The test **does not** prove host-level air-gapping, resistance to Docker/kernel escapes, immutable image provenance beyond digest selection, future workload containment or absence of secrets in a future experiment. It is one probe on one runner environment, not a security certificate.

## New static contract

`cl10f_container_boundary_contract.py` accepts only an exact, review-only Docker command **declaration**. It requires network none, read-only root, dropped capabilities, non-root UID, no privilege escalation, explicit CPU/memory/PID ceilings, `--pull=never`, and a full immutable image digest. Mounts, environment injection, host networking, privileged flags, extra commands and all mutable tags fail closed.

This validator **never executes** its declaration. Deliberately it cannot append an AI workload command; a later separately reviewed and authorized implementation must explicitly bind a pinned scientific program and frozen inputs into the proven boundary and verify that its constraints remain effective. Static source checks alone can never grant `GO`.

## Unresolved work

1. Independently review threat model and whether this container boundary can satisfy the prior `verified_no_network_host` requirement; do not silently redefine host isolation.
2. Build a future, independently reviewed fail-closed execution adapter **without enabling it**, including exact allowed executable, no credential propagation, input/output mounts and provenance, and runtime boundary validation.
3. Finish CL-10A science gates: train-only calibration, frozen data digests, numeric tolerance and source-event instrumentation.
4. Obtain independent scientific/security signoff and Marcus's separate **exact** bounded execution authorization before any AI work.

**Current decision: NO_GO.**
