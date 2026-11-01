# CL-5K — disconnected decision adapter for future generator counters

## Goal

CL-5J provides an event counter ledger. CL-5I identified the generator's
short-circuit rejection decision:

`if remaining[label] == 0 or features in used: continue`

CL-5K builds a **standalone, disconnected adapter** that accepts an already
computed remaining class quota and a lazy duplicate-check callback. It reports
`class_full`, `duplicate`, or `accepted` to the CL-5J ledger without
handling any real features, RNG or datasets.

## Intended future call placement (NOT wired here)

After the existing RNG feature proposal and XOR label, an eventual patch
could call the adapter once. A proposed duplicate lookup would reflect the
existing membership test, but only when the class quota is not full. The
generator's original accepted-record operations (adding to `used`, appending
the record, and decrementing the class quota) must remain in the same order.

The adapter itself **must not** become the authoritative sampling decision
without explicit equivalence tests. Its return value only describes the
provided predicates and increments counters.

## Contract and risks

1. `remaining_for_label == 0` skips the duplicate callback entirely.
2. With a positive quota, the duplicate callback is called exactly once.
3. A non-boolean duplicate result is rejected; invalid quotas fail before
   beginning a proposal.
4. If a callback raises or returns an invalid value after the attempt begins,
   the ledger remains pending and intentionally cannot be finalized as a
   successful measurement. A future integration must handle this failure
   safely rather than publishing partial counts as complete.
5. All input predicates are supplied by a caller; this is not evidence of
   accurate measurements from the actual generator.

## Required before connecting the generator

- Independent code review of a proposed generator instrumentation patch.
- Demonstration that RNG calls, accepted vectors, ordering and digests are
  unchanged with instrumentation toggled. Such a demonstration would require
  separately authorized **data generation**.
- Immutable reporting of failed as well as successful partitions.
- Keep model experiment training behind its own distinct explicit approval.

No production generator source is modified. No data generation, model
inference/training or hardware energy measurement takes place in CL-5K.
H1 remains `NO_GO` and all nine scientific readiness gates remain open.
