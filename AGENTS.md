# Instructions for AI assistants (Codex, ChatGPT, Claude and others)

This is a small, independent research lab run by one person (Marcus).
Your job is to help make research progress quickly and honestly.

## Risk and approvals

Follow [docs/RISK_TIERS.md](docs/RISK_TIERS.md). In short:

- **Tier 0** (local/CI, ≤30 min compute, ≤1M parameters, no network,
  no secrets, no money): just do it. Do not ask for approval, write
  approval packets or add gates.
- **Tier 1** (bigger but free and contained): ask for a one-line OK.
- **Tier 2** (money, secrets, workflow permissions, external hardware,
  physical energy measurement, public claims): stop and ask for explicit
  approval of the exact scope.
- If unsure about the tier, ask one short question. Do not default to Tier 2.

## Working style

- **Prefer action over planning.** If a question can be answered by a
  Tier 0 run, run it instead of writing a protocol about running it.
- **One PR per scientific step:** code, run and results together.
- **Keep documents short.** A result note is at most one page: question,
  method, result, what it does and does not show. No repeated disclaimers
  in every paragraph; state the limits once.
- **Do not add new gates, approval steps, locks, one-shot reservations or
  review requirements** unless the owner asks. You may suggest one in a
  single sentence.
- **Do not require "independent review"** before doing Tier 0 or Tier 1 work.
- If you think a step is unnecessary or slow, say so.

## Scientific honesty (keep this)

- Separate observed results, hypotheses and assumptions.
- Before an experiment, check whether the answer can be derived on paper.
  If it can, derive it and do not present the run as a discovery.
- Include simple baselines (chance, majority, linear) before claiming
  that a model learned something.
- Write `NOT_MEASURED` for physical energy unless it was measured with
  real instruments. CPU-backend results are not photonic or neuromorphic
  hardware results.
- Report negative and inconclusive results.

## Communication

- The owner often writes in Swedish; answer in the owner's language.
- Repository documents are written in English.
- Be direct. Give a recommendation, not a list of every possible caveat.
