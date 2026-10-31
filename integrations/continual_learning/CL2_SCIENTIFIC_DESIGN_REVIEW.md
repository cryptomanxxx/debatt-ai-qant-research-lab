# CL-2 scientific design review — pre-experiment audit

**Status: open question, not a completed or authorized experiment.** This review documents a measurable confound in the candidate generator from #162. Nothing here authorizes research compute.

## Finding: task-dependent XOR rules are inherently inconsistent

With `T1` using bits (0,1), `T2` (2,3), `T3` (4,5), and `T4` (6,7), two different tasks assign different labels to **50% of all possible binary vectors under a uniform reference distribution**. The new `exact_uniform_rule_conflict` proves that fact by exhaustive enumeration of the four relevant bits. The candidate currently withholds task identity at inference, so perfect simultaneous performance on all four incompatible labeling functions is impossible for a single deterministic global classifier. **Do not interpret resulting accuracy loss by itself as catastrophic forgetting.**

The 50% figure is *not* the conflict rate under the class-balanced candidate sampler, nor a measured benchmark outcome. Splits have no identical feature vectors, but that cannot remove incompatibility of the underlying rules.

## Finding: the proposed activity probability is not realized sparsity

The generator balances labels 50/50 by rejection sampling, changing the effective feature distribution. Actual spike fractions must be measured per partition. The new `audit_candidate_splits` checks labels, disjointness, task order and observed activity. These values remain **algorithmic input statistics**, not energy.

## Design alternatives requiring scientific review

1. **Recommended primary diagnostic:** one global label rule shared by all four tasks, with four documented input distribution shifts, to make retention interpretable without intrinsic label contradictions. Include carefully selected controlled distribution shifts and prevent trivial task leakage.
2. **Secondary interference stress test:** retain the contradictory task-specific XOR rules, but *explicitly label* the study as an impossibility/interference experiment; compare against task-ID-aware upper bound or task-specific heads.
3. **Ablations:** keep input sequences and parameter budgets identical across dense/sparse models and independently vary update locality and temporal spiking dynamics. Current CL-1 models are mathematically identical and cannot show an anti-forgetting benefit.
4. **Quality gate:** preregister minimum above-majority accuracy and per-task acceptance criteria, seed count and holdout methodology before fitting any model. Otherwise a low-access but inaccurate model is not an energy efficiency success.

**Recommendation:** do not freeze `cl2_protocol_draft.json` or authorize model training yet. Choose an identifiable primary task design, obtain scientific review, then pin generator SHA, protocol commit and compute budget.
