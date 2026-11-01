# CL-6J — Read-only SHA-256 source verification (NO TRAINING)

**Status: NO_GO. No model construction, inference, training, data generation or hardware access.**

CL-6I checked whether declared provenance fields *look valid*. CL-6J adds a narrowly bounded check against **real bytes** of the four committed CL-2 model-source files:

- `cl2_mlp_baseline.py` (dense)
- `cl2_local_mlp.py` (local)
- `cl2_sparse_forward.py` (sparse)
- `cl2_shared_index_forward.py` (shared index)

It first validates the CL-6I manifest shape, then reads only those four fixed files using paths defined by the existing CL-6B audit. SHA-256 mismatches, missing files or malformed declarations fail closed. Caller-provided path names are never accepted.

A successful source check establishes **only** that the four files currently on disk matched the four declared digests. It does not authenticate the supplied Git commit identifier, attest that files belong to that commit, pin imported dependencies, validate the source-to-data relationship, or establish that declared input data exists or has matching bytes. The returned result retains `NO_GO` and flags these remaining dimensions as unverified.

A separate, future and appropriately authorized process must resolve immutable Git objects, lock dependencies and independently verify any actual experimental data and paired-run artifacts. No dataset is downloaded or generated here. The two systems under comparison also need separate numerical equivalence and scientific quality evidence.

All nine CL-6A scientific and authorization gates remain OPEN.
