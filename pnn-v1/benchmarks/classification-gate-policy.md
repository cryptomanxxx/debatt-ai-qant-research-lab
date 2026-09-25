# Classification gate policy

For fixed-size classification benchmarks, research decisions at exact accuracy boundaries must use **integer correct-prediction counts**, not floating-point mean accuracy comparisons.

Example: over 10 seeds × 100 TEST cases, “within 1 percentage point” is represented as “no more than 10 fewer correct predictions out of 1000”.

Mean accuracy remains a reporting metric. It should not be the gating representation when the same decision can be expressed exactly with counts.

This policy follows boundary cases observed in PNN-v1 Exp006 and Exp021, where mathematically exact thresholds were vulnerable to binary floating-point representation.

Implementation rule for new PNN experiments (Exp022 onward): proposal validation rejects mean-Q.ANT-accuracy inequality gates. Aggregate correct counts should be computed from integer per-seed correct counts whenever possible.
