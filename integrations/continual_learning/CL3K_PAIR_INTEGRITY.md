# CL-3K: pre-experiment paired result integrity contract

**NO_GO — no research training, run, data read or energy measurement.**

A future continuous-learning comparison should contain ten records: five fixed
paired seeds × the two preregistered model arms. The pure validator rejects:

- missing or duplicate seed/arm combinations;
- inconsistent experiment protocol, code commit, generator, runtime,
  accounting scope or approval identifiers;
- different dataset digests or update counts within a seed's two arms;
- unexpected identical dataset digest across separate paired seeds.

Each record must already pass CL-3J's *format-only* validation. CL-3K
checks internal consistency but does not establish that digest strings refer
to real files, that recorded approvals are authentic, or that both arms
actually trained with the stated data. Such authenticity and run-lineage
checks require an independently reviewed future runner, pinned immutable
artifacts and Marcus's separate explicit human compute approval.

The tests use entirely fabricated manifests. They are **not empirical
results**, do not instantiate any model, and do not change CL-3A's NO_GO
status. Consistent metadata is necessary for a fair paired study, but it
is not sufficient evidence of equal runtime, energy or scientific validity.
