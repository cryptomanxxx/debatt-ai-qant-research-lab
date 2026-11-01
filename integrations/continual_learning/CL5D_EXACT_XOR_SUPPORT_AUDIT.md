# CL-5D — exact XOR feature-support audit (NO DATA GENERATION / NO TRAINING)

## Scientific question

Does the synthetic experiment design have enough distinct binary vectors to
meet its globally disjoint, class-balanced train/test quotas? And how does
the amount of *available* support depend on input sparsity?

CL-5D answers **only** the combinatorial support question, without sampling
vectors. The existing generator's rejection sampler has an additional
distributional and computational cost that remains unmeasured.

## Derivation

For `N = 16` binary features and `Y = X0 XOR X1`, fix the total
number of active bits to `k`. The other `14` bits are unrestricted.

- `Y = 1`: choose exactly one of the first two bits, then choose `k-1`
  of the other 14, giving `2 * C(14, k-1)` possible vectors.
- `Y = 0`: both XOR bits are zero or both are one, giving
  `C(14, k) + C(14, k-2)` possible vectors.
- Out-of-range binomial coefficients count as zero.

Summing across weights gives `32,768` distinct vectors per class and
`65,536` possible feature vectors overall.

For four tasks, each asking for 128 training and 64 test examples, equal
class quotas demand `4 * (128+64)/2 = 384` **distinct examples in each
class across all tasks**. Therefore `384 <= 32,768` passes the
necessary global-count check.

This is **not** a sufficiency proof for the current rejection sampler:
all nonzero Bernoulli rates have full theoretical support, but the probability
mass is strongly concentrated at particular activity weights. Conditional
class quotas, finite sampling attempts and global deduplication can cause
rare-event delays or failure even when combinatorial capacity is sufficient.

## Evidence limits

The audit is pure integer arithmetic and unit tests; it creates no feature
vectors, dataset records, random draws, experiment jobs or model updates.
The `NO_GO` state and all nine H1 scientific gates remain unchanged.
These are static feature-space counts, not empirical observations or evidence
of actual learning, forgetting or lower energy use.

Later research should examine exact conditional mass, cross-task uniqueness
pressure and rejection rates before approved data-generation experiments.
Training remains separately prohibited without explicit user authorization.
