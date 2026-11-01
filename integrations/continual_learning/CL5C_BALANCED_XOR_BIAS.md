# CL-5C — exact balanced-XOR proposal bias (NO DATA GENERATION / NO TRAINING)

## Research concern

CL-5B validates supplied datasets but must not assume that an empirical activity
rate equals its nominal Bernoulli proposal parameter. The actual generator
rejection-samples for exactly balanced XOR labels **and** uniqueness across all
tasks and partitions. Class balancing by itself shifts activity whenever
`p != 1/2`; the global uniqueness condition can shift it further.

## Analytical control (before global uniqueness)

Let input bits be iid Bernoulli(`p`), `N=16`, and label `Y=X0 XOR X1`.

- `Pr(Y=1) = 2 p (1-p)`.
- `E[X0+X1 | Y=1] = 1`.
- `E[X0+X1 | Y=0] = 2 p² / (p² + (1-p)²)`.
- With forced equal numbers of labels 0 and 1:
  `E[active bits] = 14p + (1 + 2p²/(p²+(1-p)²))/2`.
- Compare this with the unconditioned proposal expectation `16p`.

The helper uses `fractions.Fraction` to avoid float-rounding ambiguity.
At `p=1/8`, the proposal expects `2` active bits; equal XOR class
balance alone changes that expectation to `2.27` active bits.
At `p=1/2`, the two expectations coincide at `8` active bits.

## Why this is not a prediction of the generated datasets

This derivation does **not** model uniqueness rejection or finite candidate
exhaustion. The four tasks share a global used-feature set, so subsequent tasks
also depend on earlier draws. Therefore these numbers are *analytical reference
values*, not observations, guaranteed means, corrected generator outputs or
evidence of an energy saving.

## Safety and evidence

The implementation and unit tests use exact arithmetic only. They do not import
or call `generate_shared_rule_candidate`, sample random values, produce
datasets, run inference/training, or measure physical energy. The H1 protocol
remains `NO_GO` with all nine scientific gates open. Empirical distribution
tests and dataset hashes await an explicitly approved generation protocol;
experiment training requires a separate explicit approval.
