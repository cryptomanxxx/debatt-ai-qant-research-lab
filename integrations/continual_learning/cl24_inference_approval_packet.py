"""CL-24 authorization packet preflight. Strictly NO MODEL IMPORT/INFERENCE.

No executable inference path is present in this revision. A future, separately
reviewed PR and Marcus's exact approval are required before model execution.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from cl19_single_sample_forward_review import validate_review_only as audit_cl19
from cl22_model_source_lock import audit_current_files as audit_cl22

HERE = Path(__file__).resolve().parent
SPEC = HERE / "cl24_inference_approval_packet.json"
MODEL_MODULES = (
    "cl2_mlp_baseline", "cl2_local_mlp",
    "cl2_sparse_forward", "cl2_shared_index_forward",
)
EXPECTED_SAMPLE = {
    "dataset": "integrations/continual_learning/frozen_data_candidate/cl13/seed-101.json",
    "seed": 101, "task": "T1", "partition": "train", "index": 0,
    "features": [0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    "label": 0, "heldout_access": False,
}
EXPECTED_MODELS = {
    "arms": ["local_hybrid", "sparse_local_hybrid", "sparse_shared_index"],
    "architecture": [16, 8, 2], "parameters_per_model": 154,
    "initialization_seed": 101, "instances_max": 3,
    "forward_calls_max": 3, "weight_updates_max": 0,
    "training_steps_max": 0, "loss_gradient_calls_max": 0,
}
EXPECTED_RUNTIME = {
    "proposed_network": "none", "pids_limit": 32, "memory_mib": 128,
    "cpus": 1, "timeout_minutes": 10, "read_only": True,
    "no_new_privileges": True, "capabilities": "ALL_DROPPED",
    "uid_gid": "65534:65534", "paid_runners_allowed": False,
    "final_model_image_digest": None, "runner_image_digest": None,
    "container_includes_sample": False, "host_boundary_verified": False,
    "artifact_provenance_approved": False,
}
EXPECTED_APPROVALS = {
    "numeric_tolerances_approved": False, "scientific_signoff": False,
    "security_signoff": False,
    "marcus_exact_inference_authorization_reference": None,
}
EXPECTED_OUTPUTS = {
    "physical_energy": "NOT_MEASURED",
    "max_artifact_mib": 1, "observations_recorded": 0,
}


class InferenceNotAuthorized(RuntimeError):
    pass


def attempt_inference(*_args, **_kwargs) -> None:
    """Hard stop. There are no import, model construction or forward paths."""
    raise InferenceNotAuthorized("CL-24 NO_GO: separate inference approval required")


def validate_review_only(packet: dict) -> dict:
    if (type(packet) is not dict
            or packet.get("kind") != "CL24_ONE_SAMPLE_INFERENCE_AUTHORIZATION_PACKET"
            or packet.get("version") != "cl24-review-v1"
            or packet.get("decision") != "NO_GO"
            or packet.get("review_only") is not True
            or any(packet.get(k) is not False for k in (
                "execution_workflow_created", "model_import_authorized",
                "inference_authorized", "training_authorized",
                "energy_measurement_authorized"))):
        raise ValueError("CL-24 NO_GO: approval or execution drift")
    if packet.get("sample") != EXPECTED_SAMPLE or packet.get("models") != EXPECTED_MODELS:
        raise ValueError("CL-24 NO_GO: scientific scope drift")
    if packet.get("runtime") != EXPECTED_RUNTIME:
        raise ValueError("CL-24 NO_GO: runtime envelope or blocker drift")
    if packet.get("approvals") != EXPECTED_APPROVALS:
        raise ValueError("CL-24 NO_GO: signoff drift")
    if packet.get("outputs") != EXPECTED_OUTPUTS:
        raise ValueError("CL-24 NO_GO: output or observation drift")
    if any(any(name in module for name in MODEL_MODULES) for module in sys.modules):
        raise ValueError("CL-24 NO_GO: forbidden model module imported")
    cl19 = json.loads((HERE / "cl19_single_sample_forward_candidate.json").read_text(encoding="ascii"))
    previous = audit_cl19(cl19)
    locked = audit_cl22()
    if (previous["decision"] != "NO_GO"
            or previous["forward_calls_executed"] != 0
            or locked["decision"] != "NO_GO"
            or locked["source_files_pinned"] != 10):
        raise ValueError("CL-24 NO_GO: existing data/model source gates failed")
    return {
        "decision": "NO_GO",
        "scope": "ONE_SAMPLE_THREE_FORWARDS_PROPOSAL_ONLY",
        "source_files_verified": 10,
        "models_imported": 0,
        "forward_calls_executed": 0,
        "training_steps": 0,
        "physical_energy": "NOT_MEASURED",
        "remaining_blockers": [
            "final_model_image_and_input_sample_not_built_together",
            "immutable_final_image_and_runner_identity",
            "independent_host_boundary_security_assessment",
            "independent_numeric_tolerance_and_scientific_signoff",
            "marcus_separate_exact_inference_authorization",
        ],
    }


def audit_current_files() -> dict:
    return validate_review_only(json.loads(SPEC.read_text(encoding="ascii")))


if __name__ == "__main__":
    print(json.dumps(audit_current_files(), sort_keys=True))
