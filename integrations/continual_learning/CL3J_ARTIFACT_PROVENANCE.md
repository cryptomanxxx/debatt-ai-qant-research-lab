# CL-3J: future experiment result artifact contract (NO TRAINING)

This PR introduces only a **pure schema validator** for hypothetical
future results. No result files are generated, downloaded or saved.

A future validated record must include:

- SHA-256 hashes of the approved protocol, generator source and dataset;
- the exact 40-hex Git commit of experiment code;
- full Python patch version, fixed model arm and seed, and a bounded update count;
- separate complete five-checkpoint/four-task accuracy and balanced-accuracy
  matrices;
- the declared source-level operation-accounting scope;
- independently verifiable identifiers for *scientific review* and
  *separate human compute authorization*.

**Critical scientific limitation:** This validator verifies only *format*.
An invented SHA or approval ID that matches the expected syntax can pass.
Therefore a syntactically valid record proves neither provenance nor that
training was approved/executed. A future reviewed runner needs immutable
artifacts, hash matching, verified approvals, train/test lineage and
repeatable metadata before any result can be trusted.

Unit tests use explicitly fictional manifest values. CL-3A remains NO_GO,
and this PR does not introduce a runner, physical-energy claims, training,
inference, network calls, or output writes.
