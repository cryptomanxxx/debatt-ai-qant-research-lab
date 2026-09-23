# Research job governance

Before an AI researcher is allowed to propose compute work, the orchestration
layer enforces a guarded job lifecycle.

## Lifecycle

`proposed -> approved -> running -> completed | failed`

Version 2 executes only jobs whose committed definition has status
`approved`. Proposed jobs cannot consume compute.

## Guardrails

The runner does not accept an arbitrary shell command. A job names a committed
Python experiment under `experiments/` or `scripts/`. The executor rejects
path traversal, missing files, unsupported runners, unapproved jobs and budgets
outside hard limits.

Current hard ceilings are 10M parameters, 500 candidates, 100 epochs, 60k
training samples and 10k test samples. Individual jobs can declare smaller
budgets. These fields establish the control plane; experiments will
progressively be refactored to consume and enforce their declared budget
directly.

## Audit trail

Successful runs create an execution record containing the job ID, source Git
commit, GitHub runner and GitHub run ID. Scientific result files and execution
records are committed back to the repository.

## Next step

AI Researcher v1 should initially be allowed to create only `proposed` job
definitions. Human approval changes a reviewed job to `approved`; only then
can the compute runner execute it.
