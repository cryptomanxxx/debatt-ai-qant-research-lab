# CL-4B — instrumented source-loop probe (NO TRAINING)

This deliberately narrow preparatory probe iterates over **actual Python loop indices** corresponding to the CL-2 16→8→2 dense and sparse forward traversal patterns. It records source-level event visits (first and second layer weight terms, input visits, index reads/writes, hidden-nonlinearity sites). It **does not** instantiate a model, access learned weights, run classification inference, generate data, update parameters, or execute a training task.

The corresponding source event counter in CL-4A is a future observation contract; CL-4B probe output is **not yet an experimental observation of the actual CL-2 model implementation**. The planned causal comparison remains local dense versus local sparse under the same update rule, contingent on independent equivalence and heldout-quality review.

At two active bits of 16, the isolated probe visits 144 forward weight terms in the dense path and 32 in the sparse path (16 first layer + 16 output layer). The sparse probe additionally visits 16 input bits to build the index list and performs 16 active-index reads and 2 index writes; the dense probe visits 128 first-layer input bits. These counts are **observed in the probe's loops**, not measurements of CPU instructions, cache misses, real DRAM traffic or physical power. They are expected to match the narrower first/second layer algebra but need independent integration into actual CL-2 paths before model-level conclusions.

The full 0–16 activity sweep is verified with deterministic unit tests. There is **no timing benchmark**: Python instrumentation changes runtime, and no timing-overhead baseline has yet been reviewed. All nine CL-3Y gates remain open; CL-3X and CL-4A remain **NO_GO**. This PR provides no experiment runner and no authorization to train.

To validate only the probe's source-loop behavior:

`python -m unittest -v tests.test_cl4b_source_event_probe_contract`
