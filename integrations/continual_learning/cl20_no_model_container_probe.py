"""CL-20 NO-MODEL runtime boundary probe. No neural model imports.

Proves only this concrete container instance cannot establish an outbound
TCP connection. Not a proof of host air-gapping or future model execution.
"""
from __future__ import annotations

import errno
import json
import os
import socket
import sys
from pathlib import Path

MODEL_NAMES = ("cl2_mlp_baseline", "cl2_local_mlp", "cl2_sparse_forward",
               "cl2_shared_index_forward", "cl10b_gated_h1_runner")


def verify_no_model_runtime() -> dict:
    if any(any(name in module for name in MODEL_NAMES) for module in sys.modules):
        raise RuntimeError("CL-20 NO_GO: forbidden model module import")
    if os.geteuid() != 65534:
        raise RuntimeError("CL-20 NO_GO: not unprivileged user")
    if not Path("/opt/research/integrations/continual_learning/cl19_single_sample_forward_candidate.json").is_file():
        raise RuntimeError("CL-20 NO_GO: model proposal missing")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(2)
        code = sock.connect_ex(("1.1.1.1", 443))
    if code not in (errno.ENETUNREACH, errno.EHOSTUNREACH, errno.EPERM, errno.EACCES):
        raise RuntimeError(f"CL-20 NO_GO: unexpected network outcome {code}")
    return {
        "decision": "NO_GO",
        "scope": "THIS_CONTAINER_NO_MODEL_NETWORK_PROBE_ONLY",
        "tcp_connect_result_errno": code,
        "network_egress_denied_in_this_container": True,
        "host_network_isolation_independently_approved": False,
        "future_model_image_digest_pinned": False,
        "future_model_inference_authorized": False,
        "models_imported": 0,
        "training_steps": 0,
        "inference_steps": 0,
        "physical_energy": "NOT_MEASURED",
    }


if __name__ == "__main__":
    print(json.dumps(verify_no_model_runtime(), sort_keys=True))
