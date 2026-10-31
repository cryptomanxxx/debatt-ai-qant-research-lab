# CL-3M: predeclared analytical reference controls

Status: **data-contract checks only, NO_GO for experiment training.**

For the synthetic global XOR-label candidate:

- Exactly half of every train and heldout partition belongs to each class,
  so a constant-class classifier is expected to achieve **50%** accuracy
  on this prebalanced sample by construction.
- The known global XOR function agrees with 100% of generated labels **by
  construction**; this is a mathematical oracle/data-integrity check, not a
  trained classifier result or a meaningful independent learning baseline.
- These reference values are invariant across the four tasks and five fixed
  paired seeds, provided the generator keeps its present balance and label
  rules. Class-conditional sparsity still varies as documented in CL-3G.

Run: `python -m integrations.continual_learning.cl3m_reference_controls --check-data`.

The tool generates only ephemeral in-memory synthetic fixtures to verify
their fixed rules. It does **not** fit models, compare learning methods,
evaluate generalization, measure physical energy or authorize any experiment.
A separate scientific review must determine additional nontrivial
baselines, suitable task validity and evidence standards before training.
CL-3A remains NO_GO.
