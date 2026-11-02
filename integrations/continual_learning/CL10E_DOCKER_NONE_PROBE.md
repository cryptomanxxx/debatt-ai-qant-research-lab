# CL-10E — Proposed manual Docker network-none probe (NO_GO)

This PR adds a **manual-only** GitHub Actions workflow; no automatic push/PR/schedule trigger, no model import, training, inference, dataset generation, energy measurement or experiment authorization. The probe is **not run by this PR**. Review its exact steps and GitHub billing/quota settings before a manual invocation.

## Test boundary

The standard `ubuntu-24.04` GitHub host remains connected to the Internet. In a setup step, Docker pulls the pinned BusyBox image by immutable manifest digest. The runner launches a fixed Python HTTP server on its local Docker bridge interface. The probe then:

1. Confirms that a restricted BusyBox container with normal `bridge` networking reaches this **host-controlled** HTTP service. Without this positive control, the negative test is inconclusive and must fail.
2. Launches an equivalent container with `--network none`, read-only filesystem, dropped Linux capabilities, no privilege escalation, 32-PID/128 MiB/one-CPU caps, and confirms the same HTTP fetch **fails**.
3. Fails closed if network-none unexpectedly reaches the test service. Its result is plain workflow log text; no secrets, source data or generated results are mounted into either container.

This is stronger evidence than simply comparing network-namespace identifiers or relying on a DNS failure. It is still **container-scoped evidence only**. It does not prove the host has no Internet access, that a future model stays in the same container boundary, or that no kernel/Docker escape is possible. The positive bridge container is intentionally permitted to access only a controlled host service for this test; it is not a model workload.

## Remaining blockers

- Independent review of the isolation design and exact probe image/commands.
- Actually observing and preserving an approved manual probe run on the target runner.
- Binding any later training process to this same fail-closed, no-network container boundary with independently reviewed immutable source/input provenance.
- Separately verifying host and Docker runtime trust, absence of credentials, image and supply-chain provenance, enforcement of resource ceilings, durable artifacts, and scientific gates.
- Separate explicit authorization from Marcus for any exact experiment. Codex approval and a green network probe do **not** grant that authority.

**Decision remains NO_GO.** No Google e2-micro is needed to review or prepare this candidate.

## Documentation

- [Docker: none network driver](https://docs.docker.com/engine/network/drivers/none/)
- [GitHub: GitHub-hosted runners](https://docs.github.com/en/actions/reference/runners/github-hosted-runners)
