"""CL-23 no-model candidate image probe. Imports audit/probe only, never models.

This is NOT a verification of kernel/host air-gapping, and final image
identity and model-execution security remain unapproved.
"""
from __future__ import annotations

import json
import sys

from cl20_no_model_container_probe import MODEL_NAMES, verify_no_model_runtime
from cl22_model_source_lock import audit_current_files


def verify_review_only() -> dict:
    if any(any(name in module for name in MODEL_NAMES) for module in sys.modules):
        raise RuntimeError("CL-23 NO_GO: model was imported")
    locked = audit_current_files()
    if locked["decision"] != "NO_GO" or locked["source_files_pinned"] != 10:
        raise RuntimeError("CL-23 NO_GO: source lock failed")
    network = verify_no_model_runtime()
    if network["decision"] != "NO_GO" or not network["network_egress_denied_in_this_container"]:
        raise RuntimeError("CL-23 NO_GO: network probe failed")
    return {
        "decision": "NO_GO",
        "source_files_pinned": 10,
        "network_egress_denied_in_this_container": True,
        "models_imported": 0,
        "model_inference_authorized": False,
        "training_authorized": False,
        "physical_energy": "NOT_MEASURED",
        "final_model_image_approved": False,
        "host_isolation_independently_approved": False,
    }


if __name__ == "__main__":
    print(json.dumps(verify_review_only(), sort_keys=True))
