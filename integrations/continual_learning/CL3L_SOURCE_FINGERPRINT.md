# CL-3L: review-only source fingerprint preview

This stage provides a deterministic, read-only SHA-256 preview of selected
CL-2/CL-3 implementation and protocol files. It supports comparing code
versions during later independent scientific review.

Command: `python -m integrations.continual_learning.cl3l_source_fingerprint --preview`.

The tool hashes an explicit allowlist and computes a canonical aggregate
digest. It does **not**:

- pin or authenticate a Git commit, branch, tag or signed release;
- include the complete Python runtime, third-party packages or every
  transitive file influencing an experiment;
- establish actual dataset-content digests, approved scientific hypotheses
  or any independently verified human compute authorization;
- generate data, instantiate models, train, score, save artifacts, contact
  hardware or assert energy measurements.

A future approved experiment must freeze **the entire relevant dependency
closure**, environment and data, verify source hashes against a Git commit,
and bind both scientific and separate human compute approvals to that exact
version. CL-3A remains **NO_GO**.
