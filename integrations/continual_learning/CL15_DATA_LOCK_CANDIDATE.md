# CL-15 H1 data identity candidate — not yet frozen

The accompanying JSON pins the exact five ordered-record SHA-256 values confirmed in the CL-14 audit run: https://github.com/cryptomanxxx/debatt-ai-qant-research-lab/actions/runs/41433518948
It also pins the CL-13 generation run, source head commit and ZIP artifact digest.

**This is a durable receipt, not a durable dataset.** The underlying GitHub Actions artifact expires on 2026-11-16, so a later explicitly reviewed storage change must preserve the original records before expiry. Storing only digests would make reproducibility dependent on re-generation and might lose evidence.

This candidate makes no independent scientific claim of data quality, no inference/training authorization and no energy claim. The NO_GO state continues, and formal data-freeze approval remains separate.
