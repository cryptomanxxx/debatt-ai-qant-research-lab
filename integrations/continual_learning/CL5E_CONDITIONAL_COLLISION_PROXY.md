# CL-5E — exact class-conditioned proposal concentration (NO DATA / NO TRAINING)

## Motivation

CL-5D proved that each XOR class has 32,768 possible 16-bit vectors. It did
**not** show that these vectors are equally likely under the generator's
Bernoulli proposal distribution. For small nominal activity probabilities,
proposals concentrate on low-Hamming-weight vectors, making the size of the
support an optimistic proxy for how easily a uniqueness constraint can be met.

## Exact computation

For a bit-vector `x` with `k` active bits under iid Bernoulli(`p`) proposals,
its probability is `p^k (1-p)^(16-k)`. CL-5D supplies the exact count of
vectors for each label and active-bit count.

For either XOR label `c`, let `P_c = Pr(XOR=c)`. The exact probability that
two **independent, with-replacement proposals conditioned on that same label**
match is:

`collision_c = sum_{x: XOR(x)=c} [Pr(x) / P_c]^2`.

A uniform distribution over the 32,768 label-compatible vectors would yield
`1/32768`. The diagnostic computes both class-conditioned collision
probabilities as exact rational numbers for T1..T4. It also checks that
class probabilities sum to one and that collision probabilities are at least
the uniform support reference.

## Interpretation limits

These are *independent two-proposal matching probabilities*, not measured
duplicates from the actual generator. The real algorithm rejects duplicates
across previous classes, splits and tasks, and applies finite-attempt stopping.
This analysis does not model that history, provide an expected runtime or
prove that a run succeeds. A small theoretical collision probability is not
proof of efficient sampling; a large one is not a measured failure rate.

No vectors, dataset records, random draws, model predictions, training steps,
memory traffic or energy measurements are produced. The existing H1 `NO_GO`
decision and nine readiness gates are unchanged. Actual data-generation
experiments and all model training require their respective separate approvals.
