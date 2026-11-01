# CL-5L — opt-in instrumented generator candidate (NO DATA / NO TRAINING)

## Scope

This PR adds a **separate opt-in generator implementation**,
`generate_shared_rule_candidate_with_counters`. It uses the CL-5J
counter ledger and CL-5K decision adapter to report proposal, class-full,
duplicate and acceptance counts per task partition.

The existing `generate_shared_rule_candidate` is **not modified**.
The new function has no automatic caller and is not connected to GitHub
Actions. Source-only tests inspect its AST; they do not invoke either
generator or construct records.

## Design assumptions to review

- Seed recipe, per-task `random.Random`, 16 Bernoulli bit draws per
  candidate, XOR labels, class balance, global uniqueness and partition
  order are intended to match the original function.
- Proposals are counted only after passing the early completed-quota break.
- The class-full predicate short-circuits duplicate membership checks.
- Acceptance retains the original `used.add`, `records.append` and
  quota-decrement order.
- Attempt cap remains one million iterations for each partition.
- Only a fully completed call returns records and per-partition reports.
  On capped failure it raises, so **partial diagnostic reports are not
  exported** in this first candidate. A later reviewed design should
  preserve bounded failure diagnostics without changing source semantics.

## What is *not* verified

This PR does not establish data equivalence. The function has not been
executed and no actual datasets have been generated. Static similarity
cannot prove that ordering, random streams or digests are identical.

Before using this instrumented implementation for research, obtain a
separate **data-generation-only** authorization and a reviewed protocol
that compares original and instrumented results for the same precommitted
seeds, quotas, canonical record digests, and failure behavior. Rejection
statistics must be checked for conserved attempt counts and proper
class/duplicate reason attribution.

There is no training, inference or energy measurement. All nine H1
scientific readiness gates remain open and the protocol remains `NO_GO`.
Model experiment training requires an independent explicit approval,
even if future data-only validation succeeds.
