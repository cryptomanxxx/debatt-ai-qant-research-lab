# CL-3B safe offline preflight scaffold

**Status: NO-GO for model training.** This adds a *read-only* preflight and future result-schema checks, **not** an executable training runner. No CL-2 study, cloud call, vendor SDK, energy measurement, or dataset generation runs in this PR.

## What is implemented

- `python -m integrations.continual_learning.cl3b_preflight --preflight` prints deterministic JSON derived solely from the committed CL-2 draft and CL-3A review. It validates that readiness is `NO_GO`, all eight blockers remain open, and `compute_approved=false` / `runner_enabled=false`. It does not instantiate models.
- `--execute` is **hard-disabled** and returns an error without calling `step()` or accessing hardware.
- Maximum *proposed*, not used, budget: **5 seeds × 2 arms × 512 updates = 5,120 sample updates**; planned held-out prediction positions: **5 seeds × 2 arms × 5 checkpoints × 4 tasks × 64 held-out records = 12,800**. This count excludes training-forward work and does not estimate time, cost, or energy.
- `cl3b_result_contract.py` defines a strict 5×4 accuracy matrix shape and mandatory provenance fields, including scientific review and compute approval identifiers. It performs no file writes. The unit fixture of 0.5 accuracy is solely a schema fixture, **not an experiment result**.
- A focused CI unit test checks the no-go state, budgets, disabled execute branch and schema validation. No full-run tests exist or are triggered.

## What remains before an actual experiment

1. Resolve/approve all scientific CL-3A blockers, in particular task validity and learnability calibration *without touching final held-out labels*, generator source hash and paired data digests.
2. Specify a bounded offline runner *in a separately reviewed PR*, with no network access, safeguards against test-label use for updates, deterministic checkpoint logging, failure preservation, and strict runtime budget ceilings.
3. Bind source code/protocol SHA-256 and data provenance to an **independent scientific review**, then request **separate human compute authorization** following [research governance](../../docs/current-systems-and-protocols.md). Do not rely on a self-editable JSON flag as approval.
4. Only after approval may a future runner actually instantiate the models for sequential fitting and create signed/hashed result records. **PR merge and green tests are never approval.**

The current preflight is safe to inspect manually but does not grant or implement experimental execution.
