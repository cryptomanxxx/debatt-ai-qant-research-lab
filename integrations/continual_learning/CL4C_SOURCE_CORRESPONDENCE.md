# CL-4C — verify CL-4B source-loop correspondence (NO TRAINING)

This offline AST audit inspects source text, not running models. It detects specific changes to the dense first-layer weight traversal, the sparse first-layer active-index traversal, active-index construction, and the common output-layer traversal. It checks that CL-4B's illustrative two-active-bit event totals remain 144 versus 32.

**Scope:** these are intentionally narrow source-shape assertions, not a proof of numerical equivalence, accuracy parity, code coverage, runtime behavior, CPU instructions, memory transfers or joules. An implementation could preserve these AST patterns yet change surrounding behavior; an independently reviewed model-path instrumentation and equivalence analysis is still necessary. AST drift fails closed and requires a human to re-review the probe.

The audit uses `ast.parse` and never instantiates CL-2 models or calls model `step`/`predict`/`_forward`. Tests intentionally mutate in-memory source strings to verify that stale probes cannot pass after selected loop changes. No network, synthetic dataset, learning or experimental inference is performed.

The nine CL-3Y gates remain open and CL-3X/CL-4A decisions remain **NO_GO**. Passing CI is not permission to run an H1 experiment.

Offline check: `python -m unittest -v tests.test_cl4c_source_correspondence_contract`.
