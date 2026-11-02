# CL-20 — Candidate Python container network probe (no model)

This is a **no-model** probe in the proposed Python runtime family.
It copies the research sources and the previously frozen-candidate H1 data
into a Python image, but its entrypoint imports **no AI models** and
executes neither forward inference nor training.

GitHub Actions builds the image and runs it with explicit Docker restrictions:
network=none, all Linux capabilities dropped, no-new-privileges,
read-only filesystem, 32 PID ceiling, 128 MiB memory, one CPU and
UID/GID 65534. The process tests only whether an outbound TCP socket
to a fixed public IP is denied; successful denial establishes a
**container-specific** observation, not GitHub runner or host air-gapping.

Important outstanding blockers:
- The Python base image is a moving tag, not a pinned approved digest.
- The built local image ID is not a signed, remotely immutable model image.
- No actual model import/execution or runtime numerical parity is tested.
- Host network and kernel boundary are not independently certified.
- No scientific tolerance, container budget or Marcus inference approval
  is granted by this probe.
- The separate CL-19 and H1 proposals remain NO_GO; physical energy
  stays NOT_MEASURED.

Any subsequent actual inference requires a distinct exact authorization
and reviewed image/runtime artifact; this workflow cannot launch one.
