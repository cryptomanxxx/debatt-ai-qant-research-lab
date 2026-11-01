# Post-merge CI for Continual Learning contracts

The **Continual Learning CPU Contracts** workflow now runs on:

- `pull_request` for relevant CL changes (as before).
- `push` to `main` **without a path filter**, so every merge into the default branch gets the same no-training deterministic contract check against the **actual merged commit**.
- Optional manual `workflow_dispatch`.

The single job uses read-only repository permissions, a standard GitHub-hosted CPU runner, Python compilation and the existing offline `unittest` suite. It does **not** start research/model training, run model inference, call Q.ANT/Loihi APIs, or authorize compute. This change does **not** add a `push` trigger to `Validate Research Safety Fixes`, whose broader workflow includes dependency installation and other network operations.

## Interpreting status

A green PR check records tests performed before merge; a green `main` push check records tests performed on the resulting merged commit. They are separate GitHub Actions runs, with potentially different SHAs.

The Repo Autopilot dashboard may still show a **blue/pending CI indicator for an older merged PR** if it searches for checks only on that PR's original head or only for a particular event type. The new push run does **not** retroactively modify old PR check conclusions. The dashboard would need an independently reviewed change to show `PR CI` and `post-merge main CI` as separate states, or to map the merged commit SHA to push-workflow runs; it must not claim the original PR is newly passed merely because a different commit succeeded.

### Manual verification after merge

1. Open [Continual Learning CPU Contracts](https://github.com/cryptomanxxx/debatt-ai-qant-research-lab/actions/workflows/continual-learning-contracts.yml).
2. Filter for `push` event and `main`.
3. Open the newest run and check that its commit matches the merged `main` commit and that the job concluded `success`.
4. If `main` CI fails, treat this as an actual post-merge regression; do not hide it behind a previously green PR result.

**A green CI status is technical validation only.** CL-3A/CL-3X remain NO_GO; any scientific experiment/training requires a separate human decision.
