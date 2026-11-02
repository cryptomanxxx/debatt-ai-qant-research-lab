# CL-11 — Integrated H1 handoff (NO_GO, no execution)

This consolidates the existing CL-10 proposal, train-only calibration
candidate, numerical parity candidate, static Docker boundary and
artifact-coverage auditor into one **read-only review checkpoint**.

Run from this folder: `python -B cl11_integrated_noexec_handoff.py`.
Or run the tests: `python -B -m unittest -v test_cl11_integrated_noexec_handoff.py`.
Both only read static contracts and use fabricated records in tests.
Neither imports models, constructs datasets, performs inference, launches
training, executes Docker or measures energy.

The static container validator refers only to the BusyBox digest used
by the separate manual network probe. It is **not** a model image and
does not approve runner/host isolation for future experiments.

The optional `assess_supplied_artifact_handoff(path)` checks previously
written bundle shape and checksums with CL-10D, but *never* authenticates
source events, establishes physical measurements, grants scientific
signoff, or changes the final decision from `NO_GO`.

## Open blockers before any future explicit execution proposal

1. Independently review and empirically validate the train-only
   learnability calibration without leaking heldout examples.
2. Freeze exact source dataset records and SHA-256 manifests.
3. Independently approve numeric tolerances and verify genuine model
   traces and paired parity, not fabricated traces.
4. Complete and authenticate six-component model-source event capture,
   across training and heldout inference phases.
5. Demonstrate durable immutable remote artifact provenance.
6. Independently approve an exact future model container boundary and
   runner/host network isolation; the probe does not suffice.
7. Freeze runner, model-image digest, resource budgets and inputs.
8. Obtain distinct scientific and security review/signoff.
9. Obtain Marcus's **separate, specific approval** before any actual
   training, inference or dataset generation.

Current state: no training or inference authorized, no experiment
observations, no measured physical energy (`NOT_MEASURED`).
CL-11 is an integration checkpoint, **not** an execution workflow.
