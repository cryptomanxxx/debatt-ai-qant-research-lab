# CL-5I — static generator instrumentation-site review (NO DATA / NO TRAINING)

## Goal

CL-5H preregisters future empirical dataset validation, including counts of
sampling proposals and rejections. The current generator returns only accepted
records; it does not expose proposal/rejection counts. Before writing counters,
we must identify exactly which source statements determine those events.

CL-5I inspects the generator with `ast.parse` **without importing or running
the generator entrypoint**. It compares the full seven-statement proposal loop
to a reviewed AST contract, checks the single `rng.random()` call site,
and observes the generator's 1,000,000-iteration cap per partition.

## Current decision order inside the bounded attempt loop

1. Stop early if both class quotas are filled.
2. Draw 16 independent bits using `rng.random() < probability`.
3. Compute the shared XOR label.
4. Skip if that class quota is full **or** the vector already exists in `used`.
5. Mark an accepted vector as globally used.
6. Append the accepted record.
7. Decrement its remaining class quota.

The short-circuit expression in step 4 has a material consequence for future
counter semantics. If class quota is already full **and** the vector is a
duplicate, the current code rejects it by the class-full branch first.
A later instrumented implementation should preserve this priority, report
mutually exclusive reasons, and avoid any extra `rng.random()` calls or
changes to acceptance order.

## Candidate future counters (NOT implemented in CL-5I)

- `candidate_attempts` on each actual proposal after the early quota exit.
- `class_full_rejections` if the proposed label quota was already filled.
- `duplicate_rejections_after_class_check` only when class quota is open.
- `accepted_records` on append.
- `cap_exhaustion` on unsatisfied quotas after the loop.

These are proposed event semantics, not measured or implemented counters.
An instrumented version will need separate review and tests proving identical
accepted vectors, ordering, RNG draw counts and digests with counters enabled
versus disabled. Any source change invalidates the static fingerprint and
requires review. Tests that exercise the generator will require an authorized
data-generation scope.

## Safety / scientific status

No dataset generator execution, data generation, model training/inference or
physical energy measurement takes place here. The implementation adds no
production counters or CI job that runs the dataset. H1 remains `NO_GO` with
nine open scientific gates, and separate explicit permission is required for
future generation-only validation and for any later model experiment training.
