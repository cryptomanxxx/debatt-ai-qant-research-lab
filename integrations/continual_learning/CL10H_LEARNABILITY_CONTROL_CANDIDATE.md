# CL-10H — Candidate train-only learnability control and decision rule (NO_GO)

The first H1 experiment needs a way to determine whether its proposed binary XOR task is learnable **without** inspecting the reserved heldout test data. CL-10G now defines a deterministic 96-fit/32-validation training-only split per task. CL-10H nominates an existing **capacity-matched 16→8→2, 154-parameter dense SGD** model (`TinyDenseMLP`) as a separately reviewed control and freezes a **candidate** evaluation rule.

## Proposed future control (not executed)

- Five fixed seeds: 101–105. Four tasks T1–T4 in order. Same CL-10G original training-only calibration split per seed/task.
- One pass over the 96 fit records per task, at fixed learning rate 0.05; 384 updates per seed across four tasks. No model selection, heldout tuning, retries or best-seed exclusion.
- At the end of task T4, use the **32 withheld calibration-validation records** from each of four tasks: exactly 128 binary decisions per seed.
- Propose a precommitted criterion of **at least 84/128 correct** for **every one of five seeds** (corresponding to ≥0.65625 observed accuracy and thus meeting the separate proposed 0.65 floor). If any seed fails or any record is absent, report all and stop the learnability-supported scientific claim.
- Because the CL-10G subset is class-balanced (16 positive and 16 negative validation examples per task), a constant-class prediction has exactly **0.5** accuracy *if the actual frozen records conform*. The 0.65 threshold is a proposal aligned to CL-10A's minimum final accuracy, **not** empirically validated evidence or a significance test.

The dense control is capacity-matched by architecture/parameter count, but the dense SGD learning rule is different from the local/sparse learning rules. It establishes at most a possible learnability reference; it cannot alone prove causal algorithmic equivalence or that every H1 arm learns equally well. It is entirely possible that a single pass will fail; such a failure must trigger a new independently reviewed study design **before** any heldout access, not posthoc threshold changes.

## What this PR implements

- A strict JSON review-only proposal with all authorization and validity gates explicitly unresolved.
- A pure static validator and **arithmetic-only** checker for *supplied* 20 seed/task validation counts. It rejects missing, duplicate or invalid records and calculates the candidate five-seed criterion while always returning `NO_GO`, `observations_authenticated=false`, `learnability_verified=false`.
- Fixture-only unit tests using invented integer counts. No model import, generator call, fitting, prediction, real data access or energy measurement.

A future authorized calibration run must additionally freeze exact original training data, verify heldout separation and source/record provenance, implement truthful attempt/failure ledgers and obtain independent scientific review. The approved numerical parity tolerance, source-event instrumentation and exact isolated execution boundary remain open.

**CL-10H does not authorize any AI execution. NO_GO.**
