# Q.ANT / IONOS pilot — Alpha5 hardware-validation protocol (draft)

**Status:** proposed, not approved, not executed. No Q.ANT/IONOS bare-metal access, model training, external API usage or compute dispatch is authorized by this document.

**Purpose:** Prepare a scientifically defensible, bounded inference-only hardware-validation candidate for the announced Q.ANT Native Processing Server cloud pilot (November–December 2026). Actual hardware availability, accepted workloads, software interfaces, permission to measure power, and resource limits must first be confirmed by Q.ANT/IONOS.

## Evidence-based candidate selection

**Primary: PNN-v1 Alpha5 (ECG200), local width 16, 3 readout heads, Fourier frequencies [1,2], 8,550 parameters.**

Evidence (Q.ANT **CPU software backend**, **not** physical photonic hardware):

- [Exp020 recorded result](../pnn-v1/results/result020.json): Alpha5 897/1,000 aggregate Q.ANT-cpu correct over 10 seeds × 100 canonical ECG200 TEST cases, versus Alpha4 893/1,000.
- Mean reference/Q.ANT CPU absolute logit difference: Alpha5 0.02000665646 versus Alpha4 0.03264079299, roughly 38.7% smaller; observed prediction disagreement Alpha5 0/1,000 versus Alpha4 3/1,000.
- [Alpha5 architecture](../pnn-v1/models/alpha5.md) consists of 6 independent Fourier/KAN windows (16 inputs -> 16 outputs) and 3 independent 96 -> 2 readout heads with mean logits.
- [Exp021 recorded result](../pnn-v1/results/result021.json) tested compressed width 8 and 12 variants. No smaller candidate passed its preregistered eligibility decision; the recorded decision was **retain Alpha5**. Do not promote either compression variant based on this draft.
- [Exp020 source](../pnn-v1/experiments/exp020/run.py) defines the model and software calls `calc_kan_layer_fprop` and `add_bias_fprop`.

These ten seeds reuse the same canonical 100-case ECG200 TEST split; 1,000 predictions are **not 1,000 independent patients or held-out cases**. The experiment selected Alpha5 using this benchmark. A later independent dataset or held-out evaluation is required for generalization claims.

## Critical prerequisite: model weights

The historical Exp020 script records metrics but does not archive the trained `state_dict` checkpoints. **Do not assert that a pretrained Alpha5 checkpoint exists.** A future separately approved CPU reproduction must pin source SHA, dataset checksum and license, preprocessing/label mapping, seed, toolkit and numerical package versions, and store weights plus metadata/checksums in a controlled artifact store. Do not rewrite or replace frozen Exp020 evidence.

Reproduced weights may not bit-match historical results if data/package behavior differs. Compare against preregistered tolerance and disclose mismatches.

## Proposed two-stage validation after authorization

### Stage 0 — software and compatibility (no photonic claims)
1. Ask Q.ANT/IONOS whether `calc_kan_layer_fprop` and `add_bias_fprop` map to physical NPU primitives for the pilot firmware, supported dtypes/batch shapes, payload sizes, and whether weights can be provisioned without expensive reconfiguration each request.
2. Establish a clean deployable inference-only wrapper and independent PyTorch reference; validate per-layer output shape, ordering, bias, dtype conversion, and end-to-end predicted labels on permitted test inputs.
3. Check non-sensitive artifact loading, container/driver requirements, server access method, logging, and teardown on a disposable environment.
4. Review any additional training, download, or CI compute under the repository's own human-approval controls.

### Stage 1 — physical NPU pilot (only after external access and separate approval)
Compare the **same frozen weights and samples** across:
- FP32 PyTorch reference on a specified CPU;
- Q.ANT software/CPU toolkit backend;
- physical Q.ANT NPU through the vendor-supported integration, if the required primitives are actually supported.

Measure: output/logit MAE vs reference, classification disagreements, exact correct counts, per-layer numerical drift, cold vs warm latency, median and p95 latency, throughput at stated batch sizes, host→NPU transfers, weight upload/reconfiguration time, and failure rate. Report system details and software versions.

**Energy**: include only if the provider offers calibrated server/NPU telemetry or external meter access. Explicitly identify the measurement boundary (NPU, host, full server, network, cooling), idle subtraction, power sampling, uncertainty, and the same workload on the CPU control. Otherwise report 'not measured', never infer physical watts from the CPU toolkit or marketing specifications.

## Decision gates (to finalize before execution)

- Gate 1: vendor explicitly confirms the Alpha5 operation graph can use *actual photonic compute*. If not, the hardware pilot is **not** a valid Alpha5 photonic validation; stop or revise the application and preregister a supported workload.
- Gate 2: reference/backend numeric agreement must be specified as an absolute tolerance and decision-count threshold **before** the first hardware run. Do not retroactively tune gates against observed hardware outcomes.
- Gate 3: resource budget (requests, time, CPU/NPU, downloads), privacy, security, pilot terms, data/weight retention and software licenses must be approved.
- Gate 4: publish signed/hashed run manifest, negative results, measurements and exact evaluation code through the project’s review path; keep the preexisting PNN-v1 results immutable.

## Proposed pilot application description

**Title:** Hardware validation of Q.ANT-native Fourier/KAN networks for time-series classification.

**Use case:** Independent Debatt-AI Q.ANT Research Lab has developed an ECG200 time-series classifier with Q.ANT-compatible Fourier/KAN operations. The pilot would evaluate whether the same inference graph and frozen weights can execute on a Q.ANT Native Processing Unit, and compare numerical fidelity, latency, throughput, and **measured** energy (where instrumentation is available) against software-only CPU references. The initial Alpha5 model is deliberately compact (8,550 parameters) for bounded reproducible deployment. We will respect bare-metal administration and vendor security requirements.

**No claim** that CPU toolkit compatibility guarantees physical NPU support, or that photonic superiority has already been shown.

## Governance

This document is an independent pilot planning artifact, not a new PNN-v1 experiment or CL-1 training approval. Historical Exp001–Exp030, PNN-v1 Exp001–Exp021, Dev-4 and v2/v3 protocols are unchanged. Follow [the current systems/protocol map](current-systems-and-protocols.md) and request independent scientific and compute approvals before any new training or hardware execution.
