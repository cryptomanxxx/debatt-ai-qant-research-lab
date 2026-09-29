# Current systems and protocols — documentation map

> **As of 29 September 2026.** This is a navigation map for `main`, not a new experiment protocol, research permission, data migration, or compute authorization. For a specific study, use its committed implementation, versioned machine-readable protocol, reviewed proposal/job, actual source/result records and provenance at the relevant Git revision.

## System status and source of authority

| System / scope | Status | Where to find the governing records |
|---|---|---|
| Original Toolkit discovery, `Exp001–Exp030` | **Preserved historical series**, encompassing MNIST/FashionMNIST baseline, Fourier/KAN and numerical compatibility research. Exp018 is not the current intervention. Exp030 received only an AST-equivalent readability refactor in #152. | [`experiments/`](../experiments/), [`results/`](../results/), [`research_queue/proposals/`](../research_queue/proposals/), [`research_queue/analyses/`](../research_queue/analyses/), [Exp030 audit](experiment030_readability_refactor.md). |
| PNN-v1 | **Separate active architecture-development series.** Committed numbered experiment results through PNN-v1 Exp021 at this snapshot, plus separately identified auxiliary confirmations. | [`pnn-v1/`](../pnn-v1/), [PNN overview](../pnn-v1/README.md), [count-based gate policy](../pnn-v1/benchmarks/classification-gate-policy.md), each study's exact proposal, run script and results. |
| AI Researcher v2 | **Connected proposal/review/compute pipeline.** Committed backend configuration is currently `deterministic`, `proposal-only`, `allow_paid_usage=false`; historical proposal outputs may have different provenance. | [`researcher_backend.json`](../research_queue/researcher_backend.json), [`ai-researcher-v2.yml`](../.github/workflows/ai-researcher-v2.yml), the compiler and review/compute workflows linked below. |
| AI Researcher v3 | **Separate bounded research tooling.** Offline strategy-isolated read-only Research Memory and source-pinned hypotheses; any real Groq session is separately gated. A v3 proposal, review or tool lookup is not a completed experiment. | [Research Memory](ai-researcher-v3-memory-foundation.md), [model adapter](ai_researcher_v3_live_adapter.md), [hypothesis registry](ai_researcher_v3_hypotheses_and_protocol.md), manual [`first-live`](../.github/workflows/ai-researcher-v3-first-live.yml) / [`reviewed-live`](../.github/workflows/ai-researcher-v3-reviewed-live.yml) workflows. |
| Dev-4 strategy-selection comparison | **Frozen historical benchmark.** Strategy-isolated information visibility and pinned source evidence. Do not extend/restart it from this documentation or silently share cross-strategy or future information. | Exact [`dev4_selection_protocol.json`](../research_queue/benchmarks/dev4_selection_protocol.json), [initial history](../research_queue/benchmarks/dev4_initial_history_snapshot.json), [Round 3 isolated feedback contexts](../research_queue/benchmarks/dev4_round3_isolated_contexts.json), validation scripts/tests. |
| Standalone v3 width-10/width-11 study | **Draft: unapproved, unexecuted.** Proposed seeds 401–405 are fresh training seeds, not independent held-out ECG200 data; backend remains unselected. | [`first_standalone_draft.json`](../research_queue/v3_protocols/first_standalone_draft.json), [draft explanation](ai_researcher_v3_first_protocol_draft.md). Its analysis plan is not the Dev-4 protocol or a runnable job. |

**Names and evidence scopes matter:** original `Exp021`, `PNN-v1-Exp021`, auxiliary `result_engine_*`, Dev-4 selection rounds and v3 hypotheses are not interchangeable. Do not compare/merge them simply because widths or datasets match. The publishing workflow can mirror PNN output into root `results/`: a mirror is not an independent experiment.

## Scientific protocols: do not invent a universal current gate

- For ECG200 PNN-v1/local-width confirmation, determine the *exact* reviewed protocol and result before applying a gate. The recorded 100-epoch, 100-training/100-test family has a candidate paired to a simultaneous width-16 control with preregistered seeds; the relevant 5-seed confirmation uses **candidate aggregate correct ≥ concurrent control aggregate correct − 10**. This is not a rule for every result in this repository. [PNN classification gate policy](../pnn-v1/benchmarks/classification-gate-policy.md) requires exact integer correct-count decisions instead of floating-point mean boundaries.
- Frozen Dev-4 is explicitly a *different experiment-selection comparison*: ECG200, widths 4–15 against control width 16, seeds **301–305**, 100 epochs, learning rate **0.001**, batch size **32**, Fourier frequencies **[1,2]**, and per-strategy own-history isolation with a common pinned initial snapshot. Its protocol defines five planned rounds and candidate/budget constraints; this description neither authorizes remaining work nor classifies an unverified Round 3 selection as completed compute. Consult [the protocol JSON](../research_queue/benchmarks/dev4_selection_protocol.json) for exact semantics.
- The v3 standalone draft describes prospective width 10 replication with new seeds and another width 11 hypothesis, each against fresh paired control fits **if later separately approved**. Historical control scores and gates are context only. See the [draft](ai_researcher_v3_first_protocol_draft.md) rather than projecting old success criteria onto new fits.

## Current human-review and compute path

1. [`ai-researcher-v2.yml`](../.github/workflows/ai-researcher-v2.yml) creates a versioned proposal and compiler output. A prepared proposal is not an approved job. Its `latest.json` is the last recorded state, not a standing permission.
2. [`review-ai-research-proposal.yml`](../.github/workflows/review-ai-research-proposal.yml) binds scientific review to the **exact raw proposal SHA-256** and originating workflow run; approval of scientific merits is still not compute authorization.
3. [`compute-approved-ai-proposal.yml`](../.github/workflows/compute-approved-ai-proposal.yml) checks that proposal, scientific review, compiler and dispatch still match, then waits on the separate **`qant-research-approval`** environment gate. The authorized job is materialized through PR and run against immutable code and job commits by [`qant-research-queue-v2.yml`](../.github/workflows/qant-research-queue-v2.yml). [`research_queue/job.schema.json`](../research_queue/job.schema.json) gives schema ceilings; the reviewed job may impose stricter budgets.
4. The old [`approve-research.yml`](../.github/workflows/approve-research.yml) and [`approve-research-proposal.yml`](../.github/workflows/approve-research-proposal.yml) retain **disabled** jobs (`if: false`). Their earlier direct-publication description is not current procedure. PR review/merge, passing regression tests, an offline demo or an approved v3 research session does not authorize model training or a Groq call.

## What is verified evidence?

- `proposed`, `draft`, `critical_review` and `structural_checks_only_unverified` are not completed experiment results. `approved` is not proof a run actually happened. A workflow validation success is not new scientific evidence.
- Treat [`research_queue/ai_researcher/latest.json`](../research_queue/ai_researcher/latest.json), [`compiled/latest.json`](../research_queue/compiled/latest.json) and [`human_review/latest.json`](../research_queue/human_review/latest.json) as stage-specific committed snapshots. Reconstruct a decision from actual hashes, linked IDs, immutable review history and the exact corresponding source commit, not just displayed `latest` labels.
- Dev-4 Research Memory intentionally indexes only verified completed Round 1/2 results **inside the explicitly chosen strategy**. A previously completed width-10 result elsewhere in the repository can be relevant to a global audit without retroactively becoming an observation of the frozen Dev-4 GPT-OSS selector. Exclude future rounds, other strategies' outcomes, invalid attempts and holdout data from isolated retrieval.
- `qant-cpu/software-simulation` denotes the Native Computing Toolkit software backend evaluated on GitHub CPU compute, **not photonic-hardware latency, throughput, optical behaviour or energy**. This is independent Debatt-AI research.
- If a human-readable summary conflicts with primary evidence, consult the source-pinned result, governing JSON/protocol, reviewed job and code first. Preserve historical claims and hashes; correct the narrative rather than rewriting frozen artifacts.

## Documentation guide

| Read this | For |
|---|---|
| [PNN-v1](../pnn-v1/README.md), [integer gate policy](../pnn-v1/benchmarks/classification-gate-policy.md), [selection baseline](researcher-selection-baseline.md) | Series navigation and read-only measurement methods. |
| [v3 Research Memory](ai-researcher-v3-memory-foundation.md), [hypotheses](ai_researcher_v3_hypotheses_and_protocol.md), [critical review](ai_researcher_v3_critical_review.md) | Bounded research/proposal evidence, not autonomous execution. |
| [v3 standalone study draft](ai_researcher_v3_first_protocol_draft.md) | Unapproved exploratory design only. |
| [v1 overview](ai-researcher.md), [v2 provider-boundary introduction](ai-researcher-v2.md), [old approval bridge](approval-bridge.md), [early queue](research-queue.md), [early automation](automation.md), [general governance](research-governance.md) | Earlier architecture descriptions, with historical notices; do not treat examples as today's runnable path. |
| [Methodology](methodology.md), [Exp030 AST audit](experiment030_readability_refactor.md) | Software-vs-photonic limits and what #152 did/did not change. |

## Follow-up: global experiment catalog (separate PR; NOT implemented in #153)

Build a reproducible **read-only global discovery catalog** for the full repository, while preserving separate agent permissions and Dev-4 isolation:

1. Composite stable identity: **series + experiment ID + record kind/version**; identify mirrored records and do not double-count them.
2. For each actual source capture path, Git revision/blob and hashes when independently available, proposal/protocol reference, evidence status, backend, dataset/split, seeds and recorded workflow/artifact IDs. Mark missing or only claimed provenance as unverified; do not invent any measurement.
3. Store citations/pointers to primary data; an index is a derived view, not a replacement for versioned raw results or original artifacts. Preserve historical status and chronology.
4. **Separate discovery from model access.** Global catalog browsing must not automatically give a Dev-4 selector cross-strategy observations, future-round results or holdout access. A v3 strategy-isolated API must retain its independent permissions, source verification and bounds.
5. Offline deterministic builder, fixture/schema tests, duplicate/seed-leakage checks and fail-closed provenance checks. No new API request, Q.ANT run, research-budget authorization, modification of the original Dev-4 records or automatic workflow dispatch.

Finish/review #153 first; design/implement the global catalog as the next independently reviewed change.
