# CL-9D — Reproducibility and failure reporting (NO TRAINING)

**Status: NO_GO / CONTRACT ONLY.** No model or dataset is loaded, no experiment is run, and no real result artifact writer exists yet. CL-9D defines what an independently reviewed *future* H1 CPU pilot must preserve to be reproducible.

## Future immutable run manifest

Before any model execution, freeze and independently review:

- Exact Git commit and source tree identity; hashes of protocol, dataset manifest and every seed's actual data, task split and order.
- OS/runtime/package identity and the instrumented source-event definition/units.
- Ordered four arms, paired seeds 101–105, tasks and training/heldout-inference scopes.
- Named scientific/security reviewer and a distinct, workload-specific authorization from Marcus.

These are **required future fields**, not hashes that currently exist or are verified.

## Preserve every planned run and every attempt

Each future run/attempt should have a stable ID, arm, seed, task, phase, attempt number, start/end time, `COMPLETE | FAILED | INCOMPLETE`, failure reason and digest of any real output. Maintain an append-only attempts ledger and never overwrite the first failed attempt with a successful retry.

| Future status | Observation and hash semantics |
| --- | --- |
| COMPLETE | Required measured values present, real artifact checksum verified, no failure reason |
| FAILED | Nonempty failure reason; unavailable values `null`; retain observed partials |
| INCOMPLETE | Nonempty failure reason; unavailable values `null`; retain observed partials |

An unobserved value is **not** a measured zero. If no artifact exists, the artifact digest is `null`, not a placeholder SHA.

## No favorable retries or dropped seeds

- Never silently retry a failed training step or remove a weak/missing seed after seeing outcomes.
- Any retry must be reviewed separately, logged with its attempt number and retained alongside earlier attempts.
- Changing data or software after a failure starts a **new reviewed experiment**, not an invisible continuation.
- Report the denominator of five originally proposed paired seeds even if one arm or seed fails. Quality-matched H1 savings are **INCONCLUSIVE** unless the comparison is complete and meets CL-9B's full prerequisites.
- Five seeds support descriptive effects, not a general significance claim.

## Still missing

CL-8C did **not** build an immutable artifact writer or external disk quota; CL-8E's tested GitHub-hosted namespace did not meet the isolation requirement. CL-9C specifies observation fields without generating observations. Those remain separate engineering and scientific gates. All nine CL-9A gates and the independent review/Marcus approval requirements remain open.

The [machine-readable CL-9D contract](cl9d_reproducibility_failure_policy.json) and its static tests protect these boundary statements; they are not run-log ingestion or experiment execution.

**NO NEW DATA / NO MODEL CONSTRUCTION / NO TRAINING / NO INFERENCE / NO ENERGY MEASUREMENT.**
