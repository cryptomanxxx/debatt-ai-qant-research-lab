# CL-9J — GitHub Actions H1 runner risk assessment (REVIEW ONLY)

**Decision: NO_GO.** This is a documented threat model and proposed control plan, not permission to construct models, generate datasets, run inference/training, benchmark, or measure energy. It does not change or close CL-6A/CL-9B–9I gates. In particular, `verified_no_network_host` remains **OPEN**. Neither a green CI run nor Codex approval constitutes independent scientific signoff or Marcus's separate bounded execution authorization.

## Scope and environment

Target **future** tiny H1 CPU-only source-event experiments in the public `cryptomanxxx/debatt-ai-qant-research-lab` repository. Existing CL-9I CI validates static JSON, Python contracts, and synthetic mutations of configuration values; it is **not** an H1 model run. The proposed runner class is a fresh standard GitHub-hosted Ubuntu runner with `permissions: contents: read`, no secrets, no third-party AI service calls, and no user datasets. GitHub-hosted runners are internet-connected by default: no network isolation has been verified.

This document assesses feasibility, not actual live repository settings. All deployment claims below are **proposed controls pending independent verification**. Public-repository standard runner usage may be free under GitHub's published terms, but runner availability, charges, quotas, and terms must be rechecked before a future run; no Google e2-micro is required for the present static-validation stage.

## Threats, mitigations, and residual risk

| ID | Threat / validity failure | Proposed control and evidence | Residual / disposition |
| --- | --- | --- | --- |
| J1 | Outbound access could exfiltrate a token, code, or data | No environment/organization secrets passed to jobs, read-only `GITHUB_TOKEN` permissions, `persist-credentials: false`, inventory all actions and runtime network calls | Internet egress remains possible; **OPEN**, not called offline |
| J2 | Mutable actions or downloaded dependencies change execution code | Pin actions to immutable full commit SHAs; pin package hashes and Python/runtime versions; require reviewed commit and source tree digest; avoid network installs during measured phase | Verify transitive dependencies and package cache provenance; **OPEN** |
| J3 | Untrusted PR code executes with elevated privileges | Do not use `pull_request_target` or privileged write tokens for H1; restrict dispatch/runner workflow to reviewed code; require least privilege and no secrets; independently inspect triggering permissions | Repo settings and fork behavior not yet verified; **OPEN** |
| J4 | External dataset/API calls change scientific observations | No external model APIs; freeze local input/partition bytes with per-seed SHA-256; record exact experiment input digest, order, code and environment | Dataset digests and execution-time enforcement absent; **OPEN** |
| J5 | CPU sharing, throttling and scheduling confound runtime comparisons | Source-level logical event vector is primary; report wall-time/RSS only descriptively if later approved; paired randomized/interleaved order and repeat policy require separate frozen review | No energy, platform-neutral speedup, or hardware claim; **OPEN** |
| J6 | Partial/missing runs are silently discarded | Preserve append-only complete attempt ledger, all five seeds, failure reasons, immutable manifests and null-not-zero semantics under CL-9D/9H | Artifact writer/checksum verification not ready; **OPEN** |
| J7 | Unbounded jobs consume resources or unexpectedly incur cost | Hard per-job timeout, resource ceilings, no automatic training retries, no scheduled/dispatch training entry point before approval; verify public-repo billing conditions and artifact budget | Cost not guaranteed to remain zero for every future configuration; **OPEN** |
| J8 | Egress blocking is assumed from code review or a transient check | Treat static scan and `curl`/DNS spot-checks as **insufficient proof** of isolation; obtain a documented network enforcement boundary, tested deny-by-default policy, and independent audit if isolation remains mandatory | **BLOCKING** under the current `verified_no_network_host` gate |

## Decision options for independent scientific/security review

**Option A — Retain the no-network requirement.** Use a demonstrably isolated runner or workload environment with reviewed egress-deny enforcement and auditable negative tests. A default standard GitHub-hosted runner does **not** qualify. Do not infer that an `iptables` snippet, DNS failure, job-level environment variable, or a successful test establishes full host isolation.

**Option B — Amend the requirement after separate review.** Replace the broad host-isolation gate only if independent reviewers approve a narrower, threat-specific set of enforceable controls and update the upstream protocols consistently. The narrower controls must explicitly address credential exposure, supply-chain provenance, network-dependent observations, privileged triggers, frozen local data, and reproducible measured phases. This document **does not select or approve** option B.

The existing protocol has an explicit `verified_no_network_host` unresolved gate. Until a separate reviewed contract amendment chooses and validates an option, execution is **NO_GO**. A free runner being more powerful than e2-micro is not evidence that it satisfies a security boundary.

## Staged evidence plan (no experiment execution in CL-9J)

1. **Static inventory:** identify all H1-related workflows, triggers, `permissions`, action references, package install steps, runtime subprocess/network calls, and secrets references. Record a reviewed commit SHA and evidence checklist.
2. **Supply-chain control review:** independently inspect immutable action pins, Python/environment locks, local input manifests and planned offline dependency availability. Avoid making claims about undetected network access.
3. **Runner security review:** confirm public-fork trigger behavior, least-privilege permissions and whether the proposed no-secrets requirement is enforceable. Do not silently reuse deployment credentials.
4. **Network decision:** formally select option A or B through independent review; document what is enforced, by whom and how verified. Any change to `verified_no_network_host` requires its own PR and cross-contract validation.
5. **Execution readiness (later, separate authorization):** only after all non-network scientific gates (scope, scalar weights/tolerance, calibration, provenance, artifact writer, budgets and independent signoff) are resolved may Marcus consider exact bounded authorization for a specifically reviewed job.

## CL-9J review outcome

- **Feasibility:** GitHub Actions is a plausible no-extra-server candidate for future bounded CPU experiments, subject to policy and scientific review.
- **Security:** standard GitHub-hosted runners have public internet access; network-isolation gate is **NOT SATISFIED**.
- **Scientific readiness:** no pilot has been run; zero new observations; no physical energy measured; H2 not tested.
- **Next action:** independent critique and evidence inventory, **not** enabling training.

## Primary documentation consulted (2026-11-02)

- [GitHub: Private networking with GitHub-hosted runners](https://docs.github.com/en/actions/concepts/runners/private-networking)
- [GitHub: Secure use reference](https://docs.github.com/en/actions/reference/security/secure-use)
- [GitHub: Protecting against security threats](https://docs.github.com/en/code-security/tutorials/secure-your-organization/protect-against-threats)
- [GitHub: Actions limits](https://docs.github.com/en/actions/reference/limits)
- [GitHub: Choosing the runner](https://docs.github.com/en/actions/how-tos/write-workflows/choose-where-workflows-run/choose-the-runner-for-a-job)
