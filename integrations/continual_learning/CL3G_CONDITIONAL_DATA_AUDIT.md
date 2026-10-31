# CL-3G: class-conditional activity audit (no training)

Status: **NO_GO**, scientific construct validity still pending.

For the proposed four synthetic tasks and the five fixed seeds, the CL-3G
audit reports training/heldout input activity **separately within each label**,
including the two XOR-defining input bits. It also reports the absolute
difference in conditional mean input activity between labels.

Command: `python -m integrations.continual_learning.cl3g_conditional_audit --audit`.

Why this matters: every partition is forced to 50/50 labels by rejection
sampling. The original nominal Bernoulli activity rates do **not** describe
the resulting conditional distributions. Task-specific shifts can also be
class-conditional, which could complicate the meaning of memory-efficient
learning and retention even when headline sparsity appears simple.

This is an **in-memory synthetic data diagnostic only**, without training,
model inference, writing artifacts, outside hardware, a network request or
physical energy estimates. The audit does not prove dataset quality, establish
continual learning as a real-world use case, or close any CL-3A blocker.
Independent scientific review must decide whether the realized conditional
shifts are appropriate and whether baseline/oracle comparisons are adequate.
