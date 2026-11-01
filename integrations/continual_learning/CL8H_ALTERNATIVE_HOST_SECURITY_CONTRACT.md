# CL-8H — Alternative host isolation acceptance criteria (NO MODEL / NO TRAINING)

**Decision: NO_GO. Design-only contract; no host has been approved.**

CL-8E observed a GitHub-hosted `ubuntu-24.04` fixed child network namespace failure (`OS_PERMISSION_DENIED`; [run 41343841231](https://github.com/cryptomanxxx/debatt-ai-qant-research-lab/actions/runs/41343841231)). CL-8G records why this is a limitation of that tested mechanism, not proof that every GitHub Actions isolation option fails.

## Acceptance evidence required from either candidate host

| Gate | Required independently verifiable evidence |
| --- | --- |
| Host identity | Pin OS/kernel/image and the runner trust boundary |
| Network egress | Enforce outbound default deny **outside the tested child**; reproduce a fixed non-model outbound denial with policy evidence |
| Resource ceilings | Enforce CPU, memory, process count and wall time externally; prove descendant termination |
| Filesystem | Read-only inputs, separate limited output, symlink/path escape negative tests |
| Provenance | Pin reviewed commit and input digests; verify immutable artifact checksums |
| Secrets | Verify no ambient credentials; forbid untrusted PRs from invoking privileged host controls |
| Failure policy | Missing isolation, limits or evidence => NO_GO; no unsandboxed fallback |
| Independent reviews | Security review **and** scientific review, followed by Marcus's separate explicit approval |

A successful test that sees different network namespace IDs is **not** proof of blocked outbound connectivity. A container name, self-hosted label or green Actions badge is likewise insufficient.

## Practical candidate paths (not yet approved)

1. **Managed GitHub runner with external isolation:** Investigate supported externally configured networking controls without requesting privileges that the hosted runner does not grant. Test only with a reviewed, fixed non-AI probe and an authorized, controlled test destination.
2. **Administered research host:** Use an independently configured host network policy and resource controls; isolate the CI trigger from privileged execution and from repository PR contributions. It may be a small borrowed or sponsored machine; purchasing hardware is not a prerequisite for the design review.

Neither path may run models until all gates pass. The machine-readable [CL-8H contract](cl8h_alternative_host_security_contract.json) records every gate as unverified and is checked by a deterministic unit test. The earlier [CL-8C execution envelope](cl8c_cpu_execution_envelope.json) remains unimplemented and NO_GO.

**This PR creates documentation and a pure data contract only: NO MODEL, NO TRAINING, NO INFERENCE, NO NETWORK PROBE, NO NEW DATA, NO ENERGY MEASUREMENT.**
