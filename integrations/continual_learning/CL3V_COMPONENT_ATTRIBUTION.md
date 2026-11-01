# CL-3V — Explain logical-byte costs by component (NO TRAINING)

## Research question

Why does a candidate architecture's assumed logical-byte advantage change with activity? Is the difference associated with parameter accesses, input visits or active-index handling?

This is a **decomposition of a software proxy**, not a measurement of computational energy or physical memory movement.

## Exact fixed-shape reconciliation

For the audited 16×8×2 reference, with assumed widths of 8 bytes per parameter, 1 per input visit and 2 per index visit:

- **Parameters:** read/write visit counts multiplied by parameter width.
- **Inputs:** source-loop feature visits multiplied by assumed input width.
- **Indices:** source-loop index reads/writes multiplied by assumed index width.

For every integer activity level 0..16, arm and phase, the three components must sum **exactly** to the previously reviewed CL-3E (one hypothetical training update) or CL-3N (one standalone prediction) logical-payload total.

The offline workflow summary shows parameter/input/index component tables and twelve-character proportional P/I/X bars at activity 2 and 12; the downloadable `components.json` includes the complete 136-row sweep. **Training and inference are never added together.** Bars are normalized *within each row*; numbers remain the authoritative values.

This study addresses H1 mechanism attribution only. It does not observe hardware DRAM traffic or caches, tensor storage, latency, energy, classifier accuracy, or catastrophic forgetting. No model is trained or evaluated. CL-3A remains **NO_GO** for experiment training.

## GitHub Actions runtime maintenance

GitHub [removed Node.js 20 from Actions runners on 23 September 2026](https://github.blog/changelog/2026-09-23-node-20-is-no-longer-available-in-github-actions/). The CL visualizer, offline contract CI and optional manual Pages workflow now use Node.js 24-compatible `actions/checkout@v5` and `actions/setup-python@v6`; artifact upload uses `actions/upload-artifact@v6`. These action upgrades change runner tooling, not the scientific projection formulas. The other Pages action dependencies retain their own upstream lifecycle and should be checked if they produce warnings. No Pages deployment or experimental training is triggered by this PR.
