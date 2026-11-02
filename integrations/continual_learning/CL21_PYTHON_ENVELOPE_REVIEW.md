# CL-21 — Base-image digest and no-model resource envelope (NO_GO)

The exact multi-platform Python base manifest digest observed in the
successful CL-20 no-model run is now pinned in Dockerfile.cl20-probe.
The CI probe reruns with 1 CPU, 128 MiB memory, 32 PIDs, UID 65534,
a read-only filesystem, zero Linux capabilities, no-new-privileges,
network=none and a 10-minute job ceiling.

This is **not** the final immutable model image. Its local image ID can
change as the source tree changes. Neither final published image digest
nor independently verified host isolation is established.

The one-sample future CL-19 numerical proposal remains limited to three
model instances and at most three forward calls with zero weight
updates and zero training. Actual model inference has not been authorized.
Numerical tolerances, runner image identity, immutable trace provenance,
scientific and security signoffs, and Marcus's separate exact approval
remain pending. Physical energy stays NOT_MEASURED.
