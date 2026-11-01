# CL-4 scientific evidence review — H1 readiness after CL-4A–CL-4E

**Review state:** draft for independent scrutiny. **Decision:** `NO_GO` for H1 experimental training. **No model training, dataset-generation task, heldout evaluation or hardware-energy measurement was performed by this review.**

This document assesses the narrow evidence created by merged CL-4A–CL-4E against the nine open readiness gates in [CL-3Y](cl3y_readiness_audit.py) and the preregistered draft [CL-3X](cl3x_h1_preregistration.json). It is a critical synthesis, not an approval or result from a scientific comparison.

## Research question and causal contrast

Compute Paradox **H1** asks whether sparse activity can reduce *observed computation and system costs* without unacceptable accuracy loss. The primary causal contrast in CL-3X is **local_hybrid** (dense forward, local update) versus **sparse_local_hybrid** (sparse forward, same intended local update). A dense-SGD versus local-learning comparison varies the learning rule and cannot isolate sparsity. CL-3X retains five planned seeds (101–105), four tasks (T1–T4), a 16×8×2 shape, paired checkpoints and explicit heldout-quality conditions. Those plans have **not** been executed.

## Evidence ledger

| Component | Verified narrow claim | What it cannot establish |
| --- | --- | --- |
| [CL-4A](cl4a_h1_measurement_contract.py) | Validates a future event-counter/result schema, canonical digests, missing-value and no-energy-claim rules; carries all nine open CL-3Y gates. | That observations were collected, the scope was calibrated, datasets were pinned, or any gate was closed. |
| [CL-4B](cl4b_source_event_probe.py) | Executes standalone synthetic index traversals and counts source-loop visits for dense/sparse paths across 0–16 active bits. | Actual model work, CPU instructions, DRAM traffic, inference quality, or joules. |
| [CL-4C](cl4c_source_correspondence.py) | Checks selected AST generator and enclosing-comprehension patterns against real CL-2 source; detects specified source changes. | Semantic equivalence under arbitrary edits, execution coverage, source provenance or all surrounding code behavior. |
| [CL-4D](cl4d_forward_weight_read_audit.py) | On fresh, untrained models with fixed synthetic inputs, wraps W1/W2 rows to count Python `__getitem__` events during a single actual `_forward`; compares logits numerically at those inputs. | Trained-model parity, heldout classification, local-update equivalence, end-to-end overhead, cache misses or energy. |
| [CL-4E](cl4e_overhead_contract.py) | Validates synthetic ABBA timing records and computes descriptive signed differences/ratios without measuring a clock. | Any calibrated instrumentation overhead, CPU runtime speedup or physical energy. |

At **two active inputs out of 16** in the fixed 16×8×2 unit path, CL-4D observes **128 W1 and 16 W2 indexed reads** for the dense local control, versus **16 W1 and 16 W2** for the sparse local candidate. Thus the narrow totals are **144 vs 32 weight-row indexing events** in a single forward pass. This is a factor of **4.5** in that source event count, **not** a factor of 4.5 in CPU speed, memory traffic or energy. The sparse implementation also performs an active-index scan and index visits; this accounting is not contained in the W1/W2 total. Fixed-input numerical-logit parity is a unit-test result at initialized weights, not proof of task quality after learning.

CL-4D temporarily replaces row containers and restores them in `finally`; the counted event is a Python indexed weight read, not an instrumented DRAM transaction. CL-4B counts source-loop traversal events. These measurements have different scopes and must not be added as independent measurements of the same access.

## Nine scientific readiness gates — **all remain OPEN**

| CL-3Y gate | CL-4 preparation relevant to the gate | Remaining evidence required |
| --- | --- | --- |
| `source_hash_and_dataset_digest_pin` | None; CL-4 test fixtures are not preregistered datasets. | Review generator source revision; pin per-seed global split digests and show heldout disjointness. |
| `sparse_dense_local_equivalence_review` | CL-4C checks limited source structure; CL-4D checks initialized-weight forward parity at fixed inputs. | Independently verify identical initialization, local update rule, optimizer state and model quality through authorized sequential tasks. |
| `paired_statistical_plan_review` | CL-3X specifies paired seeds and quality floor; CL-4E provides synthetic descriptive arithmetic only. | Freeze uncertainty intervals, multiple comparisons, thresholds and what is done on failed parity. |
| `event_counter_scope_and_overhead_review` | CL-4B/C/D clarify three narrow source-level scopes; CL-4E drafts an ABBA method. | Instrument all required events in actual model paths, validate enabled/disabled overhead, CPU timing scope and missing-event audit. |
| `non_test_calibration_and_learnability_review` | No new CL-4 evidence. | Approve only training-set calibration/collapse rules that cannot inspect heldout targets. |
| `energy_boundary_or_explicit_not_measured` | CL-4A/E explicitly keep energy `NOT_MEASURED`. | Independently approve either that limitation for the first H1 study or a calibrated whole-system electrical meter plan. |
| `runner_source_and_artifact_schema_review` | CL-4A defines future records; current CI runs isolated contracts only. | Review immutable source version, bounded no-network experiment runner, logs, data provenance and failure behavior. |
| `independent_scientific_review` | Codex code reviews and green contract CI exist. | Independent scientific assessment of exact preregistration, causal contrast, counter scope and risk of leakage. |
| `separate_explicit_marcus_compute_authorization` | Not supplied by any PR merge. | Marcus's separate, explicit approval after the other gates; never inferred from documentation or CI. |

These are **preparatory contributions**. Neither this synthesis nor CL-4A–E closes any gate automatically.

## Scientific risks that must not be hidden

1. **Source-event savings are not resource savings.** Python loop dispatch, index building, memory layout, cache behavior and vectorized kernels may reverse or overwhelm apparent sparsity benefits.
2. **Forward-only comparison is insufficient.** H1 requires the *whole* training and prediction pipeline to be compared at matched quality, including update writes, routing and instrumentation overhead.
3. **One fixed initialization is not continual learning.** CL-4D has no task sequence, heldout accuracy or forgetting measurement.
4. **Synthetic timing is not timing evidence.** CL-4E tests a proposed arithmetic procedure, not runtime on a real runner.
5. **Photonic/Loihi claims are out of scope.** No Q.ANT or Intel neuromorphic hardware power, optical-memory, ADC/DAC or photonic performance has been observed.

## Recommended next sequence (review-only until approved)

1. **Provenance package:** inspect existing CL-2 task generator and CL-3C/G audit without generating a new dataset; propose exact source/digest pinning and leakage criteria.
2. **Causal parity package:** review dense/sparse local step and initialization paths, identify numerical and hidden-update differences; draft tests without running the experimental task sequence.
3. **Full-scope instrumentation review:** map W1/W2 reads, update writes, input scans, index handling and nonlinearities to actual code and calibrate an overhead method; document unobserved categories as missing, not zero.
4. **Protocol/runner/analysis sign-off:** finalize heldout-quality gates, paired statistical interpretation, bounded execution and an explicit `NOT_MEASURED` energy boundary for any CPU-only pilot.
5. **Separate human decision:** only after independent review, ask Marcus whether to authorize the exact bounded H1 experiment. Until then remain `NO_GO`.

**Bottom line:** CL-4 delivers credible *engineering evidence about source-level weight accesses*, including a narrow observed sparse-vs-dense difference under fixed unit inputs. It does **not** yet deliver scientific evidence of lower continuous-learning cost, retention improvement, or reduced electrical energy.
