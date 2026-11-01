# CL-5F — class-quota proposal draw floor (NO DATA / NO TRAINING)

## Problem

The current generator uses iid Bernoulli(`p`) proposals, forces equal XOR
classes within each partition and rejects previously used feature vectors.
The class constraint alone has a cost: at low `p`, XOR=1 is much rarer
than XOR=0.

## Exact lower bound under a hypothetical uncapped stream

For `Y = X0 XOR X1`, `P(Y=1) = 2p(1-p)` and `P(Y=0)=1-2p(1-p)`.
Suppose a partition demands `n/2` accepted candidates from *each* class.
Even ignoring duplicate rejection, collecting `q=n/2` **proposals** of
class `c` requires a negative-binomial waiting time with expectation
`q/P(Y=c)`. Completion of both class quotas must take at least as long
as either one individually, so:

`E[draws to fill both quotas] >= max(q/P(Y=0), q/P(Y=1))`.

The bound is calculated separately for train and test partitions and summed
for an uncapped hypothetical task. For the CL design's `p=1/8` task,
the 128-train plus 64-test quotas imply an expected-draw lower bound of
`3072/7 ≈ 438.86` candidate proposals compared with 192 accepted records.
For `p=1/2`, the analogous bound is 192 proposals. In both cases the
true expected time to satisfy *both* quotas will generally be greater.

## Critical limitations

The **actual** generator has a finite attempt cap for each partition and
may raise an error before completion. Hence the uncapped expected waiting
time is **not** a lower bound on the number of calls observed in a capped
failed run, nor a prediction of real wall-clock latency, success probability,
or memory traffic. Deduplication and previous tasks can only complicate
successful completion. The exact expected time to fill both quotas and
the probability of passing the cap are not calculated here.

All computations and unit tests use rational arithmetic only. They do not
call the generator, produce records, execute models or measure energy.
The H1 preregistration remains `NO_GO` with nine open scientific gates.
Future empirical runs and any training require separate explicit approvals.
