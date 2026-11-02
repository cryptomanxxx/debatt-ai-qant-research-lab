# CL-19: One-sample numeric forward proposal (review only)

The next *future* numeric comparison is intentionally tiny: exactly one
archived training example from seed 101 / T1 / training index 0, and one
forward computation for each of the Local Hybrid, Sparse Local Hybrid and
Sparse Shared Index variants. It proposes comparing the intermediate
hidden vector, logits, initial weights and active-index set.

The source is pinned to the permanently archived CL-13 file and its
CL-15/CL-16 checksum audit. The selected 16 binary bits and label are
reviewed in the proposal JSON; no heldout data is used for parameter tuning.

## What is NOT authorized

No model imports, construction, inference, training, gradients, weight
updates, dataset generation or energy measurement are permitted in the
current PR, CI or CLI. Running the candidate's explicit execution function
raises immediately. No executable model workflow has been created.

The CL-10I tolerance values are **candidates, not independently approved**.
A static class structure check from CL-18 cannot prove actual numerical
equivalence. Dense SGD is not part of this sparse ablation comparison because
its update rule is intentionally different.

Before *any* future model inference, freeze exact runner/model image digests
and a CPU/memory/time budget, verify the model container's network isolation
(not merely the old BusyBox probe), approve trace provenance/immutability
and tolerances, obtain independent scientific and security signoffs, and
receive **a separate exact approval from Marcus**. The current decision is
strictly NO_GO; physical energy remains NOT_MEASURED.
