"""CL-25 in-container NO_MODEL proof for exactly one archived input."""
from __future__ import annotations

import json
import runpy
from pathlib import Path

DIR = Path("/opt/research/integrations/continual_learning")


def verify_review_only() -> dict:
    if (DIR / "frozen_data_candidate").exists():
        raise RuntimeError("CL-25 NO_GO: entire dataset must not be packaged")
    if (DIR / "cl24_inference_approval_packet.py").exists():
        raise RuntimeError("CL-25 NO_GO: execution proposal module not packaged")
    source = runpy.run_path(str(DIR / "cl25_single_input_audit.py"))
    packet = json.loads((DIR / "cl25_single_archived_input.json").read_text(encoding="ascii"))
    result = source["validate_packaged"](packet)
    previous = runpy.run_path(str(DIR / "cl23_no_model_source_container_probe.py"))
    boundary = previous["verify_review_only"]()
    if (result["decision"] != "NO_GO" or result["packaged_samples"] != 1
            or boundary["decision"] != "NO_GO"
            or not boundary["network_egress_denied_in_this_container"]):
        raise RuntimeError("CL-25 NO_GO: source or network boundary probe failed")
    return {
        "decision": "NO_GO",
        "packaged_samples": 1,
        "heldout_data_packaged": False,
        "source_files_pinned": 10,
        "network_egress_denied_in_this_container": True,
        "models_imported": 0,
        "model_inference_authorized": False,
        "training_authorized": False,
        "physical_energy": "NOT_MEASURED",
        "final_model_image_approved": False,
        "host_boundary_approved": False,
    }


if __name__ == "__main__":
    print(json.dumps(verify_review_only(), sort_keys=True))
