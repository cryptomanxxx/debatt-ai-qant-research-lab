# CL-3D: predeclared descriptive analysis contract (no training)

**NO_GO still in force**. This is analysis code operating on supplied numeric
matrices, not an experimental runner, data collector, or source of findings.

- Five fixed paired seeds: 101–105; two equally sized arms: dense SGD and
  local hybrid; four tasks; five checkpoints per arm/seed.
- Every checkpoint has accuracy for **all four** held-out task sets.
- Final accuracy is the unweighted mean over four held-out task accuracies.
- Task forgetting is max accuracy from its post-learning checkpoints minus
  final accuracy. The last task has **no forgetting estimate** (null).
- Summarize within-seed `local - dense` accuracy differences, their arithmetic
  mean and sample standard deviation, keeping all five rows. These are
  descriptive summaries; five seeds do **not** justify significance claims.
- **Conservative predeclared interpretability gate:** every seed in both arms
  must reach final mean accuracy ≥0.65 and the within-seed absolute
  between-arm difference must be ≤0.05. Failure blocks interpreting a
  source-level logical-access advantage as an acceptable accuracy tradeoff.
  It does **not** establish physical energy benefit when passed.
- No p-values, no post-hoc seed exclusions, no parameter or threshold changes
  based on held-out outcomes, and no photonic/Loihi/energy claims.

The `tests/test_cl3d_analysis_contract.py` matrices are deliberately
**fabricated contract-test inputs**, never research results.

Before real research, separate scientific review must settle construct validity,
baseline/oracle and uncertainty methodology, source-level instrument parity,
runner security and version-pinned human compute authorization. This PR makes
no changes to the CL-3A NO_GO readiness record.
