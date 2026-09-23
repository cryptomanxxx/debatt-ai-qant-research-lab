# Research Queue and Compute Orchestration

Level 2 separates the research request from the compute backend.

A job is a declarative JSON file in `research_queue/jobs/`. The GitHub Actions
queue runner receives only a `job_id`, loads the job definition, validates its
status and runner, and executes the configured command.

Flow:

```
research job -> queue -> orchestrator -> compute runner -> Q.ANT -> results -> GitHub
```

The first backend is `github-actions`. The job format deliberately includes a
runner field so future backends such as Sweden AI Factory or EuroHPC can be
added without changing the scientific experiment itself.

Example:

```json
{
  "job_id": "smoke-test",
  "status": "ready",
  "command": "python scripts/automation_smoke_test.py",
  "runner": "github-actions"
}
```

This is an infrastructure layer, not yet an autonomous AI researcher. A future
research agent can create reviewed job definitions instead of directly gaining
arbitrary access to compute.
