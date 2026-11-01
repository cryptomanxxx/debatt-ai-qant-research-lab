# CL-3T: where do sparse source-level payload savings disappear?

**NO_GO: analysis of static cost proxies only; no models run.**

Using the fixed CL-3R source-loop sweep across **0..16 active input bits**,
CL-3T reports per architecture and phase:

- the integer activity levels where hypothetical logical payload is
  *strictly lower* than the dense baseline;
- the last activity level with an advantage, and the first level where
  the advantage disappears (equality counts as no savings);
- no single crossover if the cost-advantage pattern is non-monotone.

Training and inference thresholds are calculated **separately**. The
shared-index version is optimized for the `step()` path and still shares
the ordinary sparse `predict()` path. Therefore we cannot infer the
same advantage in both phases.

The threshold depends on assumed 8-byte parameters, 1-byte input values,
2-byte active-index entries, and the current Python source-loop
accounting. It can change on real hardware due to index allocation,
cache behavior, vectorization, memory hierarchy, precision and other
unmodeled terms.

A lower logical byte proxy is **not** lower measured energy, latency or
validated learning ability. No trained accuracy, photonic/Loihi behavior,
GPU baseline or physical crossover has been measured. CL-3A remains NO_GO.
