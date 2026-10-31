# CL-3I: pre-measurement physical energy protocol

**Status: DRAFT, NO_GO. No meter, hardware execution or model training.**

Our software projections count source-level logical memory accesses and
hypothetical packed payload bytes; they are not observations of power or
energy. Any future physical comparison must explicitly predeclare:

1. **Hardware and meter:** exact CPU/GPU/photonic device, calibrated meter
   identity, accuracy, resolution, power state, software versions.
2. **Measurement boundary:** prefer measured whole-system AC wall power.
   Document whether host CPU, RAM, cooling, DAC/ADC, optical source,
   detection, data conversion and network/storage I/O are included.
3. **Baseline:** idle workload, stabilization and idle subtraction recorded
   transparently; also report total gross energy, because idle-subtracted
   net energy alone can mislead.
4. **Sampling and uncertainty:** sampling rate, clock alignment,
   measurement duration, calibration errors, aggregation and intervals.
5. **Repetition and control:** paired workloads, random order, warmup,
   environmental conditions, consistent machine utilization, uncertainty.
6. **Workload boundaries:** training, heldout inference, preprocessing and
   transfers accounted **separately**; count correct classifications and
   actual update steps from an independently approved protocol.
7. **Provenance:** code/protocol/dataset hashes, raw immutable meter logs,
   device/runtime metadata, exclusions and missing terms.

**Energy per correct classification** is meaningful only when the
classification count, quality and phase boundary match across contenders.
No photonic efficiency advantage follows from counting optical MACs without
laser, electrical I/O, memory, data converters and host overhead.

The accompanying code only validates a review-only method-draft shape
and enumerates missing preconditions. It intentionally refuses any recorded
joules, meter-ready claim or approvals. No physical energy result is
generated and no CL-3A blocker is closed. Before experiments, independent
scientific review **and** Marcus's separate human compute authorization
remain mandatory.
