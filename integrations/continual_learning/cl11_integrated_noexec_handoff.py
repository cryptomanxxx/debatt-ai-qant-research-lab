"""CL-11 integrated read-only H1 handoff. NO EXECUTION AUTHORIZED.

Combines existing proposal, scientific contracts, boundary declaration and
artifact audit. Does NOT import neural models, generate data, infer or train.
It deliberately never returns GO even for structurally complete supplied data.
"""
from __future__ import annotations

import json
from pathlib import Path

from cl10b_gated_h1_runner import validate_review_only_proposal
from cl10d_h1_coverage_audit import audit_existing_bundle
from cl10f_container_boundary_contract import (
    FIXED_FLAGS, validate_review_only_container_command,
)
from cl10h_learnability_control import validate_candidate
from cl10i_numeric_parity_audit import validate_spec

HERE = Path(__file__).resolve().parent
PROPOSAL = HERE / "cl10a_first_h1_training_proposal.json"
CALIBRATION = HERE / "cl10h_train_only_control_candidate.json"
NUMERIC = HERE / "cl10i_numeric_parity_candidate.json"
# This is ONLY the independently probed BusyBox image, never a model image.
PROBE_IMAGE = (
    "busybox@sha256:"
    "fd7dc98638c8e305f4dc34e979f1c0fdfdcaeb0fbf8fcff77ae834b6da3d7e6e"
)
BLOCKERS = (
    "train_only_calibration_not_independently_approved_or_observed",
    "dataset_records_and_digests_not_frozen",
    "numeric_tolerance_not_independently_approved_or_verified",
    "six_component_source_events_not_authenticated_end_to_end",
    "immutable_remote_artifacts_not_proven",
    "future_model_container_and_host_boundary_not_approved",
    "runner_budget_image_and_inputs_not_frozen",
    "independent_scientific_and_security_signoff_missing",
    "marcus_exact_execution_authorization_missing",
)


def _read(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if type(data) is not dict:
        raise ValueError("CL-11 NO_GO: JSON object required")
    return data


def integrated_review_only_preflight() -> dict:
    """Validate cross-component NO_GO contracts, not runtime safety."""
    proposal = _read(PROPOSAL)
    scope = validate_review_only_proposal(proposal)
    calibration = _read(CALIBRATION)
    numeric = _read(NUMERIC)
    validate_candidate(calibration)
    validate_spec(numeric)
    boundary = validate_review_only_container_command(
        ["docker", "run", *FIXED_FLAGS, PROBE_IMAGE]
    )
    if (scope["decision"] != "NO_GO"
            or boundary["decision"] != "NO_GO"
            or calibration["control"]["heldout_access"] is not False
            or calibration["candidate_decision"]["threshold_frozen_before_data_access"] is not True
            or numeric["candidate_tolerance"]["preapproved"] is not False
            or proposal["dataset"]["per_seed_dataset_sha256"] is not None
            or proposal["evaluation"]["numerical_parity_tolerance"] is not None
            or proposal["measurement"]["physical_energy"] != "NOT_MEASURED"
            or proposal["execution_budget"]["network_egress_policy"] !=
            "UNRESOLVED_NO_NETWORK_HOST_GATE"):
        raise ValueError("CL-11 NO_GO: upstream review state drift")
    return {
        "decision": "NO_GO",
        "scope": "REVIEW_ONLY_INTEGRATION",
        "planned_pairs": 20,
        "planned_training_updates_max": 10240,
        "models_constructed": 0,
        "datasets_generated": 0,
        "observations_recorded": 0,
        "network_isolation_approved": False,
        "training_authorized": False,
        "inference_authorized": False,
        "blockers": list(BLOCKERS),
    }


def assess_supplied_artifact_handoff(bundle: Path) -> dict:
    """Check existing artifact shape while forbidding scientific GO promotion.

    A complete supplied artifact still needs independent authenticity
    verification; a positive checksum alone is not an observation.
    """
    gate = integrated_review_only_preflight()
    coverage = audit_existing_bundle(bundle)
    return {
        "decision": "NO_GO",
        "scope": "SUPPLIED_ARTIFACT_STRUCTURE_ONLY",
        "coverage": coverage["coverage"],
        "heldout_records": coverage["heldout_records"],
        "training_attempts": coverage["training_attempts"],
        "source_event_records": coverage["source_event_records"],
        "source_event_matrix_complete": coverage["source_event_claim_allowed"],
        "source_event_authenticity_verified": False,
        "scientific_signoff": False,
        "training_authorized": False,
        "remaining_blockers": gate["blockers"],
    }


def main() -> None:
    print(json.dumps(integrated_review_only_preflight(), sort_keys=True))


if __name__ == "__main__":
    main()
