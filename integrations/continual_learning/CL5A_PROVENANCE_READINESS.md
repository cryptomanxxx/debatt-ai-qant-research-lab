# CL-5A — static data-provenance source readiness (NO DATA GENERATION)

This is a first *read-only* response to the CL-3Y `source_hash_and_dataset_digest_pin` requirement. CL-3X currently describes the `cl2-shared-rule-candidate-v0` generator, five paired seeds 101–105, four tasks T1–T4, 128 train and 64 test examples per task, and a proposed globally disjoint input policy. Neither the CL-3X nor the earlier CL-2 protocol has an independently approved generator-source pin or pinned per-seed dataset digest.

Existing [CL-3C](cl3c_dataset_audit.py) and [CL-3G](cl3g_conditional_audit.py) code can audit in-memory candidate data for duplicate inputs, class distribution and label-conditional activity *when separately executed*. CL-5A **does not run those generators, audit any examples, import generator modules for execution, or produce datasets**. It inspects source AST and protocol metadata and computes SHA-256 fingerprints of checked-out source files. Those fingerprints describe the exact code present **at the checkout**, not a frozen, reviewed future training dataset.

The global XOR label rule is bits 0 and 1; task shifts in activity rates and class balancing/rejection sampling can introduce class-conditional activity differences. Unique feature vectors within one seed do not establish statistical independence across seeds or prove freedom from every form of information leakage; the scientific protocol still needs independent review of its construct validity.

**What remains before the provenance gate could close:**

1. Independently approve an immutable generator-source revision and precise canonical serialization/digest method.
2. Under separate authorization for *dataset-only audit work*, produce and verify each per-seed train/test split digest, global disjointness and label correctness; keep actual split data out of logs/artifacts unless specifically approved.
3. Document task distribution, class-conditional activity and sampler rejection effects, including whether the same XOR rule yields a meaningful continual-learning challenge.
4. Bind verified content digests and source revision to the immutable experimental protocol, separately from model training authorization.

CL-5A closes **zero** of the nine scientific review gates. CL-3X stays `NO_GO`, no observed power or timing is claimed, and no experiment training has been authorized. The source fingerprint must never be confused with a dataset digest.

Read-only command: `python -m unittest -v tests.test_cl5a_provenance_source_audit_contract`.
