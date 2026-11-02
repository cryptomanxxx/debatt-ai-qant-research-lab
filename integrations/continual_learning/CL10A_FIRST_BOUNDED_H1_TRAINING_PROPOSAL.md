# CL-10A — First bounded continual-learning training experiment (FOR REVIEW, NOT EXECUTION)

**Decision: NO_GO.** This is the first concrete H1 training **candidate** after CL-9F–9M. It contains exact proposed workload dimensions and output structure, not a runnable workflow, experiment results or permission to train, generate data or infer. The companion JSON is an unsigned proposal, not an approved manifest.

## Research question

For a tiny 16→8→2, 154-parameter network on a sequential activity-shift benchmark with a common XOR(bits 0,1) label rule, does (1) sparse traversal or (2) active-index reuse change observed source-level work without unacceptable loss of heldout quality? The task shift is **not class-incremental learning**; it does not establish catastrophic forgetting or demonstrate energy savings.

## Proposed bounded workload

| Item | Exact proposal |
| --- | --- |
| CPU host | GitHub standard hosted Ubuntu runner, *only after security approval*; no paid runner/GPU |
| Model | 16 inputs, 8 tanh hidden units, 2 outputs, 154 parameters per arm |
| Arm implementations | `TinyDenseMLP`, `LocalHebbianMLP`, `SparseForwardLocalMLP`, `SharedIndexSparseForwardLocalMLP` |
| Hyperparameters | Learning rate 0.05; local hidden rate 0.01 where applicable; independent calibration review still required |
| Seeds | Exactly 101, 102, 103, 104, 105 |
| Tasks | T1 → T2 → T3 → T4, no replay, one pass, no task-ID feature |
| Source candidate | `cl2_shared_rule_candidate.generate_shared_rule_candidate`, class-balanced, globally unique binary feature vectors per seed |
| Per task and seed | 128 train and 64 heldout examples, disjoint |
| Training budget | Max 512 updates per arm/seed; 4 arms × 5 seeds = 20 arm-seed runs; **10,240 total maximum updates** |
| Heldout schedule | All four tasks evaluated at pretrain, post_T1, post_T2, post_T3 and post_T4; **25,600 planned predictions** across all arms/seeds/checkpoints if complete |
| Quality threshold | Final post_T4 unweighted T1–T4 mean >= 0.65 per arm/seed, absolute paired difference <= 0.05 |
| Repeats/retries | One specified attempt per planned arm/seed at first; no automatic retries, failures retained |
| Electricity | `NOT_MEASURED` on GitHub; separate future physical-energy campaign |

A dense SGD arm is descriptive context, not an isolated test of sparsity. The primary causal pair is local dense vs sparse local (training and heldout). The secondary pair is sparse local vs sparse shared-index (**training only**); standalone heldout inference is a no-intervention control.

## Concrete result package that the eventual authorized runner must emit

1. `manifest.json`: immutable code SHA, protocol/source-tree/environment/instrumentation SHA-256s, per-seed data digests, exact workflow/runner image, approved weights/tolerance, authorizations and budget.
2. `attempts.jsonl`: every planned arm/seed/task/checkpoint/phase attempt, timestamps, success/failure/incomplete status and reason. Null for missing values; preserve failures and retries as append-only records.
3. `heldout_accuracy.jsonl`: correct/eligible counts per arm/seed/checkpoint/task, never heldout tuning.
4. `source_events.jsonl`: raw six-field logical event counts, clearly separated train vs inference and with matching denominators. No proxy counts passed as observed events.
5. `failure_ledger.jsonl`: all incomplete/failed attempts and exceptional conditions, no posthoc exclusion.
6. `checksums.sha256`: immutable checksums of all result files.

Results should include training-side learnability and baseline checks, plus paired numeric equivalence traces at a separately preapproved tolerance. Without approved scalar weights, output only raw event vectors and descriptive deltas; **no weighted H1 direction claim**. CPU elapsed time/RSS, if reviewed, must remain separately labeled exploratory observations, never energy in joules.

## Real blockers to close before anyone starts training

This proposal intentionally leaves several values `null`:

- **Calibration and data:** CL-7C train-only validation split, class/overlap audit, train-only learnability threshold and per-seed input digests not frozen. The existing aggregate seed-101 evidence is insufficient.
- **Comparability:** independently reviewed numeric tolerance, exact source-event counter boundaries, instrumentation overhead and work weights (or a formally vector-only study design) not approved.
- **Runner and outputs:** executable future runner, immutable artifact writer, per-run duration cap, pinned image/actions/dependencies, workflow permission settings and network decision are not yet approved. The proposed 20 MiB output cap is a *candidate*, not verified enforcement.
- **Approvals:** independent scientific/security review and Marcus's **separate exact and bounded approval** remain absent.

These are scientific and execution prerequisites, not invitations to create an unlimited administrative PR series. The immediate next technical deliverable should be a reviewed frozen **data/calibration and numerical-parity specification**, then a bounded runner implementation on a separate branch. Do not add a runnable `workflow_dispatch` training job until its exact gate design is reviewed and explicitly approved.

## Review questions to unblock actual science

1. Does the XOR activity-shift data generator and train-only calibration split provide enough learnability/shift signal to make a quality-matched comparison meaningful?
2. Which exact per-operation counters and phase denominators should be frozen? If cost weights cannot be justified, should this initial study be explicitly *vector-only*?
3. Which numeric parity tolerance and tests establish that dense and sparse local arms implement the same rule, without suppressing genuine numerical differences?
4. Is the first authorized workload the complete 5-seed × 4-arm matrix, or should an independently approved narrower pilot precede it? The latter changes the protocol and cannot be silently cherry-picked.
5. Which enforceable network boundary and exact runner/manifest/approval workflow will satisfy (or formally amend) the existing `NO_GO` gate?

**No data were generated, no model was instantiated or trained, no inference was performed, and no physical energy was measured for CL-10A.**
