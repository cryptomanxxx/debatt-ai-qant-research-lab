"""CL-10F static Docker boundary checker. NO_GO / no subprocess execution.

Validates a narrow future command shape; it cannot prove host/kernel security.
Never runs Docker or imports models. A later PR must freeze a model image and
explicitly approve execution before any training/inference can be launched.
"""
from __future__ import annotations

import re

DIGEST = re.compile(r"^[a-z0-9][a-z0-9._/-]*@sha256:[0-9a-f]{64}$")
FIXED_FLAGS = (
    "--rm",
    "--pull=never",
    "--network=none",
    "--cap-drop=ALL",
    "--security-opt=no-new-privileges",
    "--read-only",
    "--pids-limit=32",
    "--memory=128m",
    "--cpus=1",
    "--user=65534:65534",
)
FORBIDDEN = (
    "--privileged", "--network=host", "--network=bridge", "--net=host",
    "--mount", "--volume", "-v", "--device", "--env", "-e",
    "--env-file", "--pid=host", "--ipc=host", "--uts=host",
    "--userns=host", "--security-opt=seccomp=unconfined",
    "--cap-add", "--entrypoint",
)


def validate_review_only_container_command(argv: object) -> dict:
    """Require one fully exact, immutable, read-only, unprivileged container envelope.

    Does not accept any command after the image: future execution arguments
    require an independent reviewed, separately authorized change.
    """
    if type(argv) is not list or len(argv) != len(FIXED_FLAGS) + 3:
        raise ValueError("CL-10F NO_GO: command shape drift")
    if any(type(x) is not str for x in argv):
        raise ValueError("CL-10F NO_GO: non-string command element")
    if argv[:2] != ["docker", "run"]:
        raise ValueError("CL-10F NO_GO: only static Docker run declaration accepted")
    if tuple(argv[2:-1]) != FIXED_FLAGS:
        raise ValueError("CL-10F NO_GO: unexpected/missing/reordered options")
    if any(x in argv for x in FORBIDDEN):
        raise ValueError("CL-10F NO_GO: privileged/network/host access forbidden")
    if DIGEST.fullmatch(argv[-1]) is None:
        raise ValueError("CL-10F NO_GO: image must be a full immutable digest")
    return {
        "decision": "NO_GO",
        "scope": "STATIC_CONTAINER_ENVELOPE_ONLY",
        "network": "none",
        "image_digest_pinned": True,
        "host_network_isolated": False,
        "runtime_boundary_verified": False,
        "models_constructed": 0,
        "training_authorized": False,
        "inference_authorized": False,
    }
