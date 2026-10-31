# CL-2 paired continual-learning experiment — review proposal

**DRAFT / NOT APPROVED / NOT EXECUTED.** This specification prepares a possible future supervised CPU experiment, not authorization for training, paid computing, Q.ANT hardware or physical energy measurements. It deliberately has no runner and no pinned generator source hash.

## Proposed design

- Models: `TinyDenseMLP` with standard backpropagation and `LocalHebbianMLP` with local hidden-layer Hebbian-like updates and supervised output rule. Both 16→8→2 tanh, 154 trainable parameters and identically seeded initial weights. They differ in optimization mechanisms, so matched parameter count **does not** establish matched effective learning capacity.
- Data: candidate shared-global-XOR-rule generator from #164, four successive task distributions T1→T4. Exactly 128 training and 64 held-out examples per task and candidate seeds **101–105**, paired across model arms; no task ID to model. Labels balanced by rejection, causing input selection bias. All splits are disjoint by binary feature vector within each seed.
- Training **if later separately authorized**: one pass over each task's 128 observations; a maximum **512 per-example updates per arm per seed**; no replay, no learning-rate selection on the test set. Model rates in JSON are *candidate choices*, not proven optimal.
- Evaluation **if authorized**: before fitting, and after each T1–T4, evaluate all four fixed holdout partitions without updates. Log complete 5×4 per-arm/seed accuracy and balanced accuracy matrices, final mean accuracy, per-task forgetting as defined in `cl2_protocol.forgetting_by_task`, and per-run provenance including software/host and source/dataset hashes.
- Sanity controls: uniform-majority baseline = 50% by construction; XOR rule oracle = 100% by definition. A successful model should be measured relative to *both*. If neither arm reaches a nontrivial quality level, the study is a baseline-failure diagnostic rather than an energy comparison.

## Prespecified interpretation gates, subject to review

- Proposed preliminary quality floor: **65% mean final balanced accuracy** on the paired held-out tasks. This number is a *candidate acceptance threshold* and must be reviewed before training, not tuned after seeing results.
- Only discuss **operation-count comparisons at matched predictive quality** when arms are within 5 percentage points of final mean accuracy and both clear the quality floor. Otherwise report a quality-versus-cost trade-off without declaring one architecture more efficient.
- Current local logical counters and dense MLP have **different coverage**; no valid cross-model access ratio exists yet. Before reporting any operation-saving ratio, build a common instrumentation specification including first-/second-layer forward/backward operations, inactive-index scans, activations, gradients, biases, memory traffic and control. Never call algorithmic counts physical watts or joules.
- No CI result, unit-test update, draft protocol, or PR merge is an approved scientific training experiment. Research governance requires separate exact-protocol scientific review and explicit human compute authorization.

## Outstanding review blockers

1. Generator `source_sha256` and immutable data provenance not pinned.
2. Scientific reviewer has not approved task validity, candidate hyperparameters, quality floor or evaluation logic.
3. No bounded trainer, metric/provenance writer or matched-scope instrumentation exists.
4. No separate compute approval; `compute_approved=false`, `runner_enabled=false`.

Next useful change: review and implement the **read-only/common measurement interface**, then freeze all versions before any approved execution.
