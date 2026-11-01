# CL-9H — Paired H1 analysis method (REVIEW ONLY; NO TRAINING)

**Decision: NO_GO.** This is an analysis specification for future independent scientific review, not a result, runtime, model, dataset or training authorization. It builds on CL-9F (consolidated protocol), CL-9G (cross-contract consistency), CL-9B (conditional outcome rules), CL-9C (measurement), and CL-9D (failure ledger). Nothing here closes an existing scientific gate.

## Estimands and comparisons

The primary estimand is the **per-seed difference in observed source-event work** between `sparse_local_hybrid` and `local_hybrid`, holding the proposed local learning rule fixed. The secondary estimand compares `sparse_shared_index` with `sparse_local_hybrid`; it is *not* pooled with the primary contrast. The SGD/dense arm is descriptive context and cannot serve as an isolated sparsity effect.

For seed `s`, phase `p` and preapproved metric `W`: `delta[s,p] = W(candidate,s,p) - W(reference,s,p)`. Negative delta means less **source-level work**; it does not demonstrate lower physical energy. Report an absolute delta and, when the reference is strictly positive, a separately labeled relative difference `delta / W(reference)`. When reference work is zero, relative difference is **undefined**, never infinity or a manufactured zero.

## Observation scope and denominator policy

- Keep `train` and `heldout_inference` separate, with matched task/checkpoint scopes across paired arms. Do not silently divide a training update count by an inference prediction count or combine them.
- Sum only **observed, complete, identically scoped** source-event counts. Six categories remain separate: logical parameter reads, writes, input visits, active-index visits, control events and nonlinear events. No proxy-only/static estimates may enter an observed count.
- A scalar weighted `W` is **not defined yet**. Unit-cost weights, scope/boundary and aggregation procedure require documented independent preapproval **before seeing outcome observations**. In its absence report raw event-vector deltas only; no support/no-advantage conclusion based on a scalar.
- Define denominators explicitly before execution: records/predictions for heldout inference; actual updates for training; record both numerator and denominator. A zero/unknown denominator makes the corresponding rate **undefined** (null); do not impute zero. Accuracy is correct predictions over observed eligible examples and must not use the work denominator.
- Quality checks must use the preregistered **final mean accuracy** definition at frozen heldout checkpoints, by seed and arm: floor 0.65; absolute paired gap at most 0.05. The exact averaging across T1–T4 and checkpoint coverage must be approved in advance. Never select a best checkpoint using heldout labels.
- Exact numerical parity (same initialization, order, dtype, output and post-update traces within independently approved tolerance) and train-only learnability are separate prerequisite gates, not inferred from similar accuracy.

## Missingness, failure, and interpretation

All five planned seeds (101–105), both causal contrasts, and every planned arm/task/checkpoint row stay in the append-only ledger. A failed/incomplete/missing row has a reason and unavailable observations as `null`, not `0`. Retries are new linked attempts; no best-of-seed choice or deletion.

**Common prerequisites, identical for positive and negative conclusions:** all five paired seeds complete with matching hashed provenance, frozen preapproved metric, same measurement scope, successful train-only learnability, quality and numerical parity gates, and no missing/excluded records. Any failed prerequisite or mixed per-seed signs forces `INCONCLUSIVE`.

Only then:
- `DESCRIPTIVE_H1_SUPPORT`: every seed has `delta < 0` for the specified contrast/phase.
- `NO_OBSERVED_H1_ADVANTAGE`: every seed has `delta >= 0` for that contrast/phase.
- Otherwise `INCONCLUSIVE`.

Mixed outcomes may still be reported descriptively with per-seed range; they do not permit a selected favorable subgroup claim. Report **both** contrasts even if one is inconclusive; no global H1 claim automatically follows from one successful contrast. Five seeds permit no generalizable p-value conclusion.

## Presentation layout for future reporting (no rows exist now)

1. Exact repository commit, protocol/data/environment hashes, approved metric definition and independent review reference.
2. Full attempt/failure ledger (including missing, incomplete and retried attempts).
3. Each arm/seed/task/checkpoint/phase: raw six-component event vector, accuracy numerator/denominator and observed-only timing/RSS if separately approved.
4. Each of the two *separate* contrasts, by seed and phase: candidate/reference, matched exposure, absolute delta, eligible relative delta or `null`, quality and numerical parity results.
5. Descriptive minimum, maximum and full signed effect list (no posthoc exclusions); state why each directional conclusion is or is not permissible.
6. Explicit limitations: `physical_energy = NOT_MEASURED`; `H2 = NOT_TESTED`; no claim about actual CPU/GPU/Loihi/Q.ANT electrical savings.

## Unresolved before any execution

1. Independently define and approve exact source-event counters and common exposure scopes.
2. Preapprove scalar work weights or explicitly restrict the study to separate six-event vectors.
3. Predefine heldout task/checkpoint quality aggregation and train-only learnability control, with frozen partition digests.
4. Approve numerical tolerance, instrumentation overhead, timing/repetition and resource budgets on a verified no-network host.
5. Obtain real external scientific signoff, then Marcus's **separate exact, bounded authorization**. Neither Codex 👍 nor green CI is such authorization.

**No new data, model execution, inference, training, runtime benchmarking or energy measurement occurs in CL-9H.**
