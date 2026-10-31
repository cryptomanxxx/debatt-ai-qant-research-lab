# CL-2 packed-payload access-volume proxy (no physical memory or energy model)

**Status:** engineering estimation from static source-level operation counts. Neither DRAM bytes nor latency, wattage, joules, photonic power, or cost-per-correct-classification are measured or established.

## Model

`cl2_payload_proxy.py` takes equal-scope `StepProfile` logical parameter reads/writes and input feature checks from `cl2_operation_accounting.py` and multiplies them by **hypothetical packed scalar widths**, with separate active-index list traffic.

Default assumptions:
- Parameters and biases: 8 bytes per logical read/write.
- Binary input check: 1 byte per selected logical check.
- Each active index list access: 2 bytes.
- Index list visits: zero for dense backprop; 1 write + 8 reads per active index for local-dense; 2 writes + 16 reads for sparse-forward/local, because its active list is computed twice and traversed for each of 8 hidden units in each path.

All values are *repeated logical access payload equivalents*, **not actual cache misses or bytes crossing a physical memory hierarchy**. A Python object uses more bytes than a packed scalar and can be cached; compiled vectorized kernels behave differently. The input-scan and index-iteration assumptions intentionally expose overhead but do not capture every allocation or branch.

## Illustrative deterministic calculations (not experimental results)

| Active binary input bits (of 16) | Dense backprop | Local dense forward | Local sparse forward |
| --- | ---: | ---: | ---: |
| 1 | 4,080 | 1,938 | 884 |
| 16 | 4,080 | 4,128 | 4,304 |

Values are **logical payload-byte proxies per proposed single update step** under the stated widths, not measured memory traffic. At full activity, the sparse implementation offers no weight traversal reduction and its extra active-index management increases this proxy. This is why operation savings cannot be assumed for all activity levels.

The `sensitivity_table` function varies assumed parameter widths among 2, 4 and 8 bytes. These scenarios do **not** account for quantization error, DRAM/cache, ADC/DAC, laser, modulators, power delivery, conversion or idle terms.

## Research interpretation and next engineering steps

1. Do not convert the proxy directly into joules or compare it to Q.ANT/Loihi benchmarks.
2. Run a separate, scientifically approved accuracy/forgetting comparison at paired seeds before even describing a quality-adjusted advantage.
3. If warranted, introduce an architecture-specific calibrated memory hierarchy and *measured* power or latency under a physically verified workload. SimPhony may help model a selected photonic/electronic architecture, but it cannot validate real hardware without calibrated inputs.
4. The sparse model currently rescans inputs to build its active list; a shared-activation-index optimization could reduce its cost but needs equivalence and accounting tests.
