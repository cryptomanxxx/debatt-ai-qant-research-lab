# CL-6H — Render-only numeric difference summary

**NO_GO: No models, datasets, training, inference or hardware measurements.**

This module converts *caller-supplied*, already computed CL-6G scalar diagnostic records for the four parameter fields into a readable Markdown summary. It does **not** generate records, import models, load weights, fetch data or execute any experiment.

- Requires an explicitly preregistered local-rule pair: `primary_sparse` or `secondary_index_reuse`.
- Requires W1/b1/W2/b2 numeric diagnostic records, checking counts, magnitudes, finite values and fail-closed scientific flags.
- Rejects incomplete data and records that claim approval or equivalence.
- Requires CL-6G's declared symmetric-relative formula and rejects contradictory zero/nonzero absolute and relative differences, including records claiming exact equality.
- Explicitly reports that learning rates, optimizer/counters, source provenance, tolerances, per-step state, model quality and energy are **not established by this partial summary**.
- Unit tests use only literal lists and the pure CL-6G numerical helper. They do not run neural networks.

This is a presentation component and **not an authorized future state-capture workflow**. It cannot close CL-6A's nine gates. Any real parameter comparisons, model initialization, inference or training require separately reviewed execution boundaries and explicit authorization.

See [CL-6G diagnostics](CL6G_NUMERIC_DIAGNOSTICS.md) and [CL-6F stepwise contract](CL6F_STEPWISE_STATE_DIFF_PROTOCOL.md).
