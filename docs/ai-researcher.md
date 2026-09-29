> **Historical v1 design, not the current connected workflow.** Start at the [current systems/protocol map](current-systems-and-protocols.md) for the active v2 scientific-review, separate compute authorization and pinned execution path.

# AI Researcher v1

AI Researcher v1 is deliberately separated from compute execution.

It reads verified experiment results and produces a structured proposal under
`research_queue/proposals/`. It cannot create an approved job and the
research workflow has no permission or mechanism to invoke the guarded compute
queue.

The first research policy is conservative: before searching a larger
architecture space, replicate promising Experiment 005 patterns across several
random seeds. This reduces the risk of treating single-seed noise as an
architectural discovery.

Human review remains the boundary between a proposal and an approved compute
job. Later versions can replace the deterministic proposal policy with a model
or agent while retaining the same proposal schema and approval boundary.
