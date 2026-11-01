# CL-9L — Proposed bounded H1 GitHub Actions runner security profile

**Status: PROPOSAL / REVIEW ONLY. Decision: NO_GO.** This profile is a set of *future acceptance criteria*, not an executable workflow, scientific approval, network-isolation attestation, model invocation, or permission to train/infer/generate data. It does not modify the CL-6A/CL-9B/CL-9H/CL-9I gates or the CL-9J/CL-9K evidence.

## Objective and scope

Propose a reproducible, bounded CPU-only GitHub-hosted H1 runner with no additional Google Cloud VM. The proposed experiment would compare the frozen H1 arms across seeds 101–105 only after separate independent scientific review and Marcus's exact authorization. The CL-9K audit found that the repository has 44 workflows, including historical jobs with write scopes and secret references, so this future runner **must be isolated by workflow design** from those jobs and their reusable workflows. A normal GitHub-hosted Ubuntu VM has public internet egress; a workflow specification by itself does not change that fact.

## Mandatory profile before a runnable workflow may be proposed

| Control | Proposed requirement | Verifiable evidence | Present status |
| --- | --- | --- | --- |
| L1 — Trigger | Dedicated H1 workflow with no automatic push, pull-request, schedule or reusable `workflow_call` training entry point. No runnable workflow created until an independently reviewed authorization design specifies how the *exact* approved run is bound to immutable input. | Reviewed event schema, dispatch policy and call graph, including all reusable workflows | NOT_IMPLEMENTED |
| L2 — Permissions | Explicit minimal `permissions: { contents: read }` for source checkout only; no `actions: write`, `contents: write`, `pull-requests: write`, `id-token: write`, package publish or Pages deployment. Set `persist-credentials: false`; prevent token from reaching experiment subprocess. | Reviewed YAML and effective job token permissions; check no inherited privileges | UNVERIFIED |
| L3 — Secrets | No workflow/organization/environment secrets, third-party model API keys, personal tokens, cloud credentials or `secrets: inherit`. Do not call historical DEV4/Groq/Q.ANT reusable research queues. | Full transitive job/step inventory and settings review by authorized owner; no secret values exposed | UNVERIFIED |
| L4 — Supply chain | Pin every nonlocal Action to reviewed immutable full 40-character commit SHA; fix interpreter/image and transitive dependency versions with hashes, preferably preprovisioned/local verified packages. No unreviewed `pip install`, remote downloads, mutable image tags or code fetched at execution time. | SHA manifest + dependency lock/hashes + independently reviewed source tree | NOT_FROZEN |
| L5 — Scientific input | Immutable code commit and protocol digest; frozen per-seed dataset bytes/digests and data ordering; approved six-field source-event definition and paired work/accuracy rules. No heldout tuning. | Signed-off manifest and matching checksums; CL-9G/I static validation | NOT_FROZEN |
| L6 — Cost and resource ceiling | Fixed CPU-only standard runner (no GPU or paid runner), hard time and disk/output caps, bounded attempt count, no automatic retry, no network cost-generating services. Prefer a small owner-approved ceiling; no duration/size is approved here. Public standard-hosted runners may be free under current terms, but **zero future cost is not guaranteed**. | Exact resource budget, quota/cost review, GitHub billing settings and immutable run parameters | UNAPPROVED |
| L7 — Provenance and failure | Append-only per-attempt ledger; include incomplete and failed seeds, null-not-zero, artifacts/checksums, environment and source hashes, no posthoc deletion or best-seed selection. | Reviewed immutable artifact writer and failure-policy conformance to CL-9D | NOT_IMPLEMENTED |
| L8 — Scientific interpretation | Phase-specific paired estimands; secondary index reuse applies to train only, heldout is no-intervention control. Accuracy at post_T4 is unweighted mean T1–T4; floor 0.65 and paired gap <= 0.05. Scalar work metric and numeric tolerance need independent preapproval. | Frozen CL-9H, CL-9I validator, independent scientific review; no early directional claim | OPEN |
| L9 — Network | **BLOCKING:** default GitHub-hosted runner is not verified offline. Option A: independently prove enforced network-isolated boundary for the exact measured workload; or option B: independently amend upstream host-isolation requirement to documented, enforceable threat-specific controls. A probe of DNS, `curl`, a transient namespace or a firewall snippet is insufficient by itself. | Separate signed network decision and enforcement validation mapped to CL-9J J8; upstream consistency review | OPEN |
| L10 — Authorization | Independent scientific and security signoff followed by Marcus's **separate, exact, bounded approval**, identifying reviewed commit, action, seeds, data digests, resource budget and permitted operations. CI and Codex thumbs-up are not approval. | Two distinct approval records plus immutable exact-run manifest verified fail-closed | NOT_GRANTED |

## Non-negotiable design rules

1. **Prepare offline where possible; measure separately.** A future setup phase must not contaminate observations. An install or network fetch within measured steps is a validity defect. This is not a claim of host-level network isolation.
2. **Never generalize from CL-8E.** Its disposable namespace smoke probe does not establish future runner isolation or authorize a model.
3. **Never inherit a prior research queue's authority.** The historical `qant-research-queue-v2.yml` has write scopes and a job executor. It is outside this H1 proposal.
4. **No scalar H1 conclusion until preregistered weights.** Raw event-vector observations (if ever authorized) alone do not justify a weighted scalar or energy-saving claim.
5. **No `GO` inference from checks.** Static checks can establish conformance of the proposed document, not environmental safety, scientific validity or user authorization.
6. **No unapproved training workflow is introduced by CL-9L.** The proposed permissions and triggers are descriptive criteria, not workflow code.

## Pre-execution gate order (proposal)

1. An independent reviewer resolves the CL-9J network-option decision and verifies the actual enforcement boundary; any upstream gate amendment requires separate reviewed PR.
2. Repo owner supplies evidence of effective Actions settings noted as inaccessible in CL-9K; decide whether to strengthen branch-review and required-check rules separately.
3. Independent scientific review freezes H1 model/data provenance, learning calibration, work definition, numeric tolerance, failure and artifact contracts.
4. Only after steps 1–3 are approved should a future dedicated bounded runner be implemented as a *separate PR*. The runner must remain disabled/fail-closed until its own security review, CI and independent signoff.
5. Marcus may then consider a new explicit authorization referencing the exact job, commit, inputs and budget. **No authorization is given in this document.**

## Decision record

**CL-9L: NO_GO / REVIEW_ONLY.** No training, inference, new dataset generation, benchmark, energy measurement, egress enforcement test, paid cloud service, or runnable H1 workflow is added. GitHub Actions remains an economical candidate, not a proven isolated or authorized experiment host.
