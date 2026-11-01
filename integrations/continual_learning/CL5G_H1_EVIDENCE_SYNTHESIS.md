# CL-5G — H1 evidence synthesis and falsification boundaries (NO COMPUTE)

## Purpose

CL-5A through CL-5F have progressively narrowed the methodological risks
in the planned H1 sparse-compute experiment. They provide useful **source
contracts and theoretical results**, but do not yet constitute evidence
that a model learns better, forgets less, performs fewer hardware operations,
or consumes fewer joules.

The read-only `cl5g_evidence_boundary_audit.py` records each stage's
supported claim and unresolved limit. It also checks the existing CL-3Y
readiness audit: the H1 protocol remains `NO_GO` with nine open gates.

## Evidence ledger

| Stage | What can be claimed | What cannot yet be claimed |
| --- | --- | --- |
| CL-5A | Reviewed generator structure and protocol literals; observed source fingerprints | A dataset was generated or its per-seed record digests were pinned |
| CL-5B | Validation functions can check **supplied** records | Real proposed datasets are reproducible, class-balanced or leakage-free |
| CL-5C | Conditional class balancing changes expected proposal activity | Generated task activity matches a given theoretical mean after uniqueness |
| CL-5D | Exact support is 32,768 vectors per XOR class | Global sequential rejection sampling meets its finite cap |
| CL-5E | Exact independent class-conditional same-vector proposal probabilities | Observed duplicates, rejection frequency, DRAM traffic or latency |
| CL-5F | Uncapped hypothetical expected-draw lower bounds for class quotas | Actual capped generator runtime, failure probability or energy use |

## Alternative explanations that must be controlled

1. **Sampling-induced sparsity:** nominal Bernoulli rates differ from the
   effective distribution after class-balanced and globally unique selection.
2. **Rare-class conditioning:** at low nominal activity, enforcing equal XOR
   labels means rejecting many more proposals, even with no duplicates.
3. **Cross-task dependence:** the global uniqueness set couples later tasks
   to earlier sampled records. Equal planned seeds do not establish independent
   samples across experiments.
4. **Quality/compute confound:** fewer logical source-loop events alone do not
   establish equal accuracy, forgetting resistance, wall time or energy.
5. **Hardware boundary:** the present software proxies exclude physical memory
   traffic, interconnect, photonic conversion overhead and measured joules.

## Recommended independent checkpoints

Before any empirical H1 claims, reviewers should freeze and approve:

- the generator source hash, version and paired seed policy;
- rules for a *separately authorized* in-memory data-generation audit,
  immutable canonical record digests, and report of empirical label/activity
  distributions by task and partition;
- overlap checks (within seed and across any claimed independent subsets),
  rejection-attempt counts, finite-cap failures and sampler limitations;
- matched architecture evaluation, statistical comparison, measurement
  boundaries, baseline quality and falsification criteria.

These are **review proposals**, not permission to execute. Codex approval
and green CI do not substitute for independent scientific sign-off.

The live H1 gates from CL-3Y remain authoritative: all nine are open.
No dataset generation, model inference/training, or energy measurement is
performed by this PR. Separate explicit authorization remains mandatory
before model experiment training.
