# CL-3O: activity-weighted prediction cost preview

**NO_GO / NO TRAINING OR MODEL INFERENCE.**

CL-3N provides a static source-loop estimate at a single assumed input
activity level. CL-3O sums those *hypothetical* estimates over the actual
binary feature activity of the predeclared synthetic dataset, separately
for each seed, task and train/heldout input partition.

This estimates source-level parameter reads, zero parameter writes,
input-feature visits, active-index overhead, first/second-layer weight
terms and logical packed payload bytes under **8-byte parameters, 1-byte
input values and 2-byte active indices**. It neither initializes nor
calls a model, and does not compute classification accuracy.

Using heldout **features** for precommitted measurement-scope accounting
does not require heldout label access. However, designing/tuning a method
after examining heldout feature distributions can still introduce adaptive
evaluation bias. This is a fixed research audit, not a reason to revise
models or hyperparameters post hoc.

The resulting totals are not CPU time, DRAM traffic, actual joules,
photonic hardware performance or evidence of successful learning.
Independent review, calibrated physical metering and separate human
compute approval remain necessary before any experiment.
