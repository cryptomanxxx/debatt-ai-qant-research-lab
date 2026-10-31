# CL-2 candidate synthetic data contract

**Not a completed dataset or authorized experiment.** `cl2_data_candidate.py` is a pure proposed generator with zero model training and no external dependencies. `cl2_protocol_draft.json` deliberately retains `generator.name=unimplemented_binary_generator` and `source_sha256=null`; this PR does not silently freeze or activate the candidate.

Proposed inputs: 16 binary features, four task IDs T1–T4 in fixed order, two shared output labels, no task ID supplied to the classifier. Each task uses a different pairwise XOR label rule and Bernoulli activity probability (0.125, 0.25, 0.375, 0.5 respectively). Seed material is namespaced and SHA-256-derived; generation order and split order are deterministic on Python 3.11.

For each task and seed, a candidate generator allocates 128 training and 64 held-out test records (balanced 50/50 labels). Candidate generation rejects duplicate feature vectors **across all tasks and both splits**; validation tests enforce this. It returns data in memory, writes no files and exposes a content SHA-256 helper.

## Scientific cautions for review

1. **Selection bias:** fixing 50/50 labels via rejection sampling modifies each task's natural Bernoulli distribution. Raw Bernoulli firing probabilities are *proposal parameters*, not guaranteed realized activity frequencies. Assess realized sparsity and train/test difficulty before approval.
2. **Task-label conflicts:** XOR rules change by task. With task ID withheld, a single global predictor may face incompatible label functions. Disjoint observed inputs do not guarantee conceptual compatibility. Include a task-ID-known upper bound, a shared-label-rule control, or reject this design if forgetting is not identifiable separately from inherent contradiction.
3. **Data leakage boundary:** records are disjoint *by binary feature vector* within each seed. That does not justify treating different seeds as independent populations for generalization claims. Never use held-out labels for updates or hyperparameter tuning.
4. **No learning claim:** dataset generation and a matching contract test neither demonstrate continual learning nor consume experimental training budget. There is no accuracy or forgetting result in this PR.
5. **Reproducibility:** a model run must later pin generator source SHA, Python version, exact seed list, task design, code commit, and all output content digests. The present draft validator expressly rejects a claimed finalized generator.

Next scientific decision: approve/change task rules and balancing, then register the agreed generator hash and *separately* authorize bounded model runs under [research governance](../../docs/current-systems-and-protocols.md).
