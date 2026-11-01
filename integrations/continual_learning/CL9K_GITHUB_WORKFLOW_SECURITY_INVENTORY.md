# CL-9K — Actual GitHub Actions security inventory (READ ONLY; NO_GO)

**Assessment date:** 2026-11-02 (Europe/Stockholm). **Scope:** public repository `cryptomanxxx/debatt-ai-qant-research-lab`, `main` as of merge `d704f91de1570ddf0d6ba7382587ebe4d5dbbc92`. This is a source/configuration snapshot, not a runtime isolation audit, independent security attestation, or experiment authorization.

**Decision: NO_GO.** No model construction, H1 training, inference, generation, benchmarking, or physical-energy measurement was performed. CL-9J's `verified_no_network_host` remains OPEN.

## Observed source-level findings

| ID | Observed evidence | Assessment |
| --- | --- | --- |
| K1 | 44 workflow YAML files under `.github/workflows/` on `main`; 42 contain a `workflow_dispatch` declaration (including reusable/legacy workflows). | Repository has broader historical research automation than the narrow CL-9I H1 contract. A dispatch declaration alone is not evidence that an H1 execution path is authorized. |
| K2 | All 44 define a top-level `permissions` block. Ten declare at least one write or `id-token: write` permission. | No blanket read-only policy across the entire repository. Not all ten are H1 jobs. |
| K3 | 14 workflow files reference `secrets.*`, including `GROQ_API_KEY` in historical AI-researcher/DEV4 workflows and `GITHUB_TOKEN` in publishing/research workflows. | Source references do not prove secrets are configured or passed to H1. Never assume an H1 experiment job has zero secrets without confirming its effective workflow. |
| K4 | Action references in the 44 workflow files use tags such as `actions/checkout@v4/v5` and `actions/setup-python@v5/v6`, not full 40-hex commit SHA pins. | External action supply-chain immutability is not fully enforced in source. Plan immutable pinning for any *future* H1 execution path, after reviewing dependency ownership. |
| K5 | `.github/workflows/cl9i-static-h1.yml` uses `contents: read`, `persist-credentials: false`, a three-minute timeout, and runs static validator/unittest files only. It has `pull_request`, `push` to `main`, and `workflow_dispatch` triggers. | Positive narrow control for CL-9I. It is not proof of internet isolation, scientific GO, or settings for unrelated workflows. |
| K6 | `.github/workflows/cl8e-linux-isolation-smoke.yml` is manual, three-minute, read-only and explicitly no-model. It runs a fixed namespace probe. | The workflow itself warns that a successful disposable child namespace is **not** proof of isolation for a future model workload. |
| K7 | `.github/workflows/qant-research-queue-v2.yml` is a reusable historical research queue with `contents: write`, `pull-requests: write`, six-hour timeout, container image and research-job executor. | Distinct prior research automation exists. It is not identified as an approved H1 CL-9 execution path; assess callers and permissions independently before any future reuse. |
| K8 | Active branch ruleset `Protect main` (ID `24041895`) targets default branch, prohibits deletion/non-fast-forward updates and requires a pull request. It specifies **0 required approving reviews** and `required_review_thread_resolution: false`. | A PR is enforced, but Codex 👍, green CI and thread closure are **not** required by this ruleset. Our review discipline is procedural, not a verified branch-rule enforcement guarantee. |
| K9 | GitHub REST branch-protection endpoint returned **403 Resource not accessible by integration**. Actions permissions-settings endpoint was unavailable through this connector. | Effective repository-level Actions permissions, permitted action policies, secret inventories, branch protection beyond the inspected public ruleset, and fork approval policies remain **UNVERIFIED**. Do not infer defaults. |

This inventory counts workflow **source files**, not reachable job executions or effective runtime permissions. Some `workflow_dispatch` strings may be in workflows that are disabled or otherwise gated; their runtime state has not been independently established.

## Provenance and review boundaries

- Source: [GitHub Actions workflow directory](https://github.com/cryptomanxxx/debatt-ai-qant-research-lab/tree/main/.github/workflows), notably [CL-9I static workflow](https://github.com/cryptomanxxx/debatt-ai-qant-research-lab/blob/main/.github/workflows/cl9i-static-h1.yml), [CL-8E probe](https://github.com/cryptomanxxx/debatt-ai-qant-research-lab/blob/main/.github/workflows/cl8e-linux-isolation-smoke.yml), [historical research queue](https://github.com/cryptomanxxx/debatt-ai-qant-research-lab/blob/main/.github/workflows/qant-research-queue-v2.yml).
- Ruleset: [Protect main](https://github.com/cryptomanxxx/debatt-ai-qant-research-lab/rules/24041895), API read of `/repos/cryptomanxxx/debatt-ai-qant-research-lab/rulesets/24041895`.
- This is a point-in-time source inspection. No user secrets were accessed, no settings were modified, and no paid cloud infrastructure was provisioned.

## Required follow-up evidence (all OPEN)

1. **Owner settings review:** GitHub Settings → Actions → General: default workflow-token permissions, allowed Actions policy, fork PR approval policy and whether immutable pinning is enforced. Requires an authorized UI/API view; absence of connector access cannot be marked PASS.
2. **Branch enforcement decision:** Decide whether to require one independent approving review, resolved threads and relevant status checks; any changes need explicit separate authorization and must consider automated Codex comments versus actual approving reviews.
3. **H1-only workflow policy:** For any proposed future H1 runner, review triggers, explicit least-privilege token, no inherited secrets, credential persistence, checkout/source digests, fixed action SHAs, frozen dependencies and strict run budget. Audit all transitive scripts and reusable workflows; a YAML-only scan cannot certify them.
4. **Network security review:** A standard GitHub-hosted runner can reach the public internet. Do not call it isolated. Independently decide CL-9J option A (genuinely isolated environment) or B (formally amended enforceable risk-based policy), then validate the chosen boundary.
5. **Scientific preregistration:** Keep all CL-6A/9B/9H scientific, artifact and independent-signoff gates open until separately evidenced. Marcus's exact bounded authorization remains mandatory for model execution.

## Conclusion

The **existing CL-9I validator** has credible minimal source-level permissions and a bounded no-model job. The **repository as a whole** contains other historical workflows with write scopes and secret references. We **cannot** claim repo-wide read-only execution, no network egress, or enforced peer-review approval. No single finding here justifies adding a Google e2-micro server; equally, free GitHub Actions alone does not close H1's isolation or scientific gates.

**CL-9K records evidence only. Status stays NO_GO.**
