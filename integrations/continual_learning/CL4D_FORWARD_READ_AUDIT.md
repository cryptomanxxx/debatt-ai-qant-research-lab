# CL-4D — actual model weight-row read instrumentation (unit-only, NO TRAINING)

This step uses fresh, deterministic, **untrained** 16→8→2 model instances solely inside Python unit tests. It wraps first- and second-layer weight rows with a transparent `__getitem__` counter, calls the actual CL-2 `_forward` function once per fixed synthetic input, then restores the original rows in `finally`.

This is different from CL-4B's isolated loop proxy and CL-4C's static AST check: it counts actual Python weight-index reads in model forward code. It still **does not** count CPU instructions, physical DRAM traffic, cache misses, energy, or execution time. It does not execute a data pipeline, training step, optimizer update, heldout task classification, or any research experiment.

Unit tests exercise a fixed two-active-bit case and 17 synthetic binary activity levels; they verify numerical agreement of dense and sparse logits within a tight absolute tolerance and compare observed W1/W2 reads against CL-4B's loop-probe counts. This does **not** establish statistical learnability, model accuracy, paired experiment parity or a causal energy advantage. Measurement overhead and input/index traffic remain outside the CL-4D read scope.

All nine CL-3Y scientific requirements remain OPEN. CL-3A/CL-3X/CL-4A remain NO_GO; **Marcus has not authorized experiment training**, and this unit audit is not such permission.

Review only: `python -m unittest -v tests.test_cl4d_forward_weight_read_audit_contract`
