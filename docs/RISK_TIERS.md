# Risk tiers for research work

**Status:** DRAFT for owner decision. Takes effect only when Marcus merges it.
**Owner:** Marcus (repository owner). Only the owner can change this policy.

## Why this exists

Early CL work treated every model operation, including a 154-parameter
pure-Python forward pass, as high-risk. Twelve milestones (CL-33 to CL-44)
gated three forward passes whose outcome was mathematically predictable,
while the unit tests ran the same models on every CI run. The caution was
well meant but misplaced: the real risks in this lab are **money, secrets,
external systems and public claims**, not arithmetic.

This policy puts approval effort where the risk is, and lets everything
else move at normal research speed.

## The three tiers

### Tier 0 — Just do it

All of these must hold:

- Runs on a laptop or a standard GitHub Actions runner.
- At most **30 minutes** of compute per run.
- At most **1M model parameters**.
- No network access during the run (installing pinned dependencies is fine).
- No secrets, API keys or paid services.
- Synthetic data generated in the run, or data already committed to the repo.

Examples: analytical calculations, unit tests, training and evaluating the
CL toy models with many seeds, density sweeps, baseline models, plots.

Requirements: commit the code and results together, record commit SHA and
seeds, and label the result **exploratory** unless it was preregistered
(see below). No approval, review packet or one-shot reservation is needed.

### Tier 1 — One-line owner OK

Anything above Tier 0 that is still free and contained, for example:

- Up to 6 hours on free GitHub Actions runners, up to 10M parameters
  (the lab's existing job ceiling).
- Downloading public datasets.
- Metered API calls (e.g. Groq) **within a monthly budget the owner has set**.

Approval: the owner writes "OK" (or similar) in the PR or issue. That is
the whole process.

### Tier 2 — Full review

- Spending money beyond the set budget, or any new paid service.
- New secrets/credentials, or changes to workflow permissions, environment
  gates or the approval workflows themselves.
- Third-party or external hardware: Q.ANT hardware, Loihi, rented cloud VMs.
- Physical energy measurement with real instruments.
- Public scientific claims: papers, preprints, posts or messages to
  companies/researchers that present results.
- Anything irreversible or that affects people outside the project.

Approval: a written proposal (one page is enough) and an explicit owner
approval for the exact scope.

**If unsure which tier applies, ask the owner. Do not default to Tier 2.**

## Exploratory vs confirmatory

- **Exploratory** work (most work) needs no preregistration. Try things,
  iterate, keep the results — including negative ones.
- **Confirmatory** work is the one run that a claim will rest on. Before
  running it, commit a short preregistration: hypothesis, primary metric,
  analysis, seeds, and what result would count against the hypothesis.
  The run itself is still Tier 0 if it fits Tier 0.

## Honest reporting (unchanged)

Keep the existing reporting discipline. It is a strength of this project:

- Separate observed results from hypotheses and assumptions.
- Write `NOT_MEASURED` instead of estimating physical energy.
- CPU-backend results are not photonic or neuromorphic hardware results.
- Report negative and inconclusive results.

## Process rules

- **One PR per scientific step**, containing code, run and results
  together — not one PR per gate.
- A milestone is done when its result is committed. A written plan alone
  is not a milestone.
- Code review (Codex, Claude, CI) is advisory for Tier 0 and Tier 1.
  A green CI run or an AI review is not required to proceed with Tier 0 work.
- No "independent scientific review" requirement before Tier 0 or Tier 1
  work. Outside critique is welcome, but is sought for finished results.
- New gates, checks or approval steps are added only when the owner asks
  for them. Existing gates that only protect Tier 0 work may be removed.

## Relation to earlier rules

When merged, this policy replaces the blanket "HARD NO_GO" for the CL track
for Tier 0 and Tier 1 work. Tier 2 keeps a full, explicit approval
requirement. Historical records remain unchanged: the consumed CL-45 one-shot
reservation is not reset or reused, and archived evidence is not rewritten.
