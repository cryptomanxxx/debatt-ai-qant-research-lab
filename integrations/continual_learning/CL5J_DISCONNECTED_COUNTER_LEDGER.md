# CL-5J — disconnected proposal counter ledger (NO DATA / NO TRAINING)

## Purpose

CL-5I identified the exact source positions for future proposal counters.
CL-5J implements **only the counter state machine** as a separate component,
without modifying, importing or invoking the generator.

The ledger receives abstract calls for a candidate proposal and its rejection
or acceptance decision. It does not require, construct, store or inspect a
feature vector. Its tests simulate decisions only; no synthetic dataset is
generated.

## Event semantics

- `begin_proposal()`: increment only after the generator's early
  quota-complete break, once per actual proposed feature vector.
- `resolve_proposal(class_full=True, duplicate_if_class_open=None)`: count a
  class-full rejection; do not claim to have evaluated the duplicate check.
- `resolve_proposal(class_full=False, duplicate_if_class_open=True)`: count a
  duplicate rejection after the class check.
- `resolve_proposal(class_full=False, duplicate_if_class_open=False)`: count
  an accepted record.
- `finish_partition(quotas_filled=...)`: reject a pending attempt, a repeated
  finish, or an incomplete partition before the attempt cap; report cap
  exhaustion separately.

The counters enforce the identity:

`candidate_attempts = class_full_rejections + duplicate_rejections_after_class_check + accepted_records`.

This is an **event-accounting invariant**, not yet evidence that the source
generator's actual event stream conforms to it.

## Required future integration evidence

Before connecting this ledger to any dataset generator, review a patch that
preserves all random draws, feature/label decisions, class-short-circuit
priority, accepted-record ordering and source versioning. After **separate
approval to generate data**, compare identical pinned-seed record digests with
instrumentation on/off, including error and cap cases. Then independently
approve immutable per-seed output artifacts. None of those experiments or
approvals happens in CL-5J.

## Scientific boundary

The ledger produces no measurements of actual rejection counts, throughput,
hardware memory traffic, training accuracy, forgetting or energy. It is not
wired to GitHub Actions as a dataset-generating task. All nine H1 scientific
gates remain open and training remains forbidden without separate explicit
authorization.
