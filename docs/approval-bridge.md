# Human approval bridge

The approval bridge converts a reviewed AI Researcher proposal into an approved
job for the guarded compute queue.

The workflow requires an explicit boolean confirmation through GitHub
`workflow_dispatch`. GitHub preserves boolean workflow inputs in the
`inputs` context, so the approval job is gated by `inputs.confirm`.

AI Researcher v1 cannot invoke this workflow and cannot change proposal status
to approved. The v1 compiler also uses an explicit proposal-to-experiment
mapping rather than accepting executable commands from proposal text.

This creates the boundary:

AI proposal -> human review/confirmation -> compiler -> approved job -> guarded queue.

The approved Experiment 006 implementation is intentionally added separately;
approval alone does not automatically start compute.
