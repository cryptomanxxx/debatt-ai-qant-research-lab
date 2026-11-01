"""CL-3Y read-only scientific readiness audit of the CL-3X NO_GO draft.

Pure metadata inspection; no training, data generation or external access.
"""
from __future__ import annotations

import json

from integrations.continual_learning.cl3x_h1_protocol_review import (
    PROTOCOL_PATH, REQUIRED_GATES, review_summary,
)

GATE_REVIEWS = {
    "source_hash_and_dataset_digest_pin": (
        "P0", "Pin reviewed generator source and per-seed global-split dataset digests",
        "CL-3C/G/M audits exist, but CL-3X dataset source/digests remain unpinned",
    ),
    "sparse_dense_local_equivalence_review": (
        "P0", "Review same-rule/local weights, initialization and forward-only contrast",
        "CL-2 model contracts and CL-3T/U/V source proxies do not prove outcome equivalence",
    ),
    "paired_statistical_plan_review": (
        "P1", "Freeze paired uncertainty, fail criteria, class/order and multiple-comparison treatment",
        "CL-3D descriptive paired summary exists; independent scientific approval missing",
    ),
    "event_counter_scope_and_overhead_review": (
        "P0", "Verify measured loop/event scope, nonlinearity and instrumentation overhead",
        "CL-3E/N/V provide projections, not observed hardware operations or memory traffic",
    ),
    "non_test_calibration_and_learnability_review": (
        "P0", "Define pre-test calibration and collapse checks without heldout leakage",
        "CL-3H provides diagnostic contracts only; learnability is not experimentally established",
    ),
    "energy_boundary_or_explicit_not_measured": (
        "P1", "Choose calibrated measurement boundary or freeze explicit NOT_MEASURED policy",
        "CL-3I is a measurement-method draft; no selected/calibrated energy meter",
    ),
    "runner_source_and_artifact_schema_review": (
        "P0", "Independently review bounded no-network runner, source hash and immutable results",
        "CL-3B is offline preflight only; no authorized executable runner",
    ),
    "independent_scientific_review": (
        "P0", "Independent scientific sign-off on exact preregistration and source versions",
        "Codex code review / green CI are not scientific experiment approval",
    ),
    "separate_explicit_marcus_compute_authorization": (
        "P0", "Obtain separate human authorization after prior gates are resolved",
        "Marcus has not authorized CL-3X model training",
    ),
}


def scientific_readiness_audit() -> dict:
    """Fail closed on unexpected gate state or protocol drift."""
    reviewed = review_summary()
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    gates = protocol["missing_review_gates"]
    if gates != list(REQUIRED_GATES) or set(GATE_REVIEWS) != set(REQUIRED_GATES):
        raise ValueError("open gates or their audit mapping unexpectedly changed")
    if reviewed["decision"] != "NO_GO" or reviewed["training_allowed"] is not False:
        raise ValueError("CL-3X unexpectedly executable")
    if (
        protocol["study_design"]["dataset_source_sha256"] is not None
        or protocol["study_design"]["dataset_digests_pinned"] is not False
        or protocol["study_design"]["primary_comparison"]["implementation_equivalence_verified"] is not False
        or protocol["measurement_plan"]["instrumentation_source_sha256"] is not None
        or protocol["measurement_plan"]["meter_calibrated"] is not False
    ):
        raise ValueError("audit needs independent approval of changed evidence state")
    rows = []
    for gate in REQUIRED_GATES:
        priority, requirement, available = GATE_REVIEWS[gate]
        rows.append({
            "gate_id": gate, "priority": priority, "status": "OPEN_NOT_INDEPENDENTLY_APPROVED",
            "requirement": requirement, "existing_evidence_and_gap": available,
            "automatically_closable": False,
        })
    return {
        "kind": "CL3Y_H1_REVIEW_READINESS_NO_COMPUTE",
        "protocol_id": reviewed["protocol_id"],
        "decision": "NO_GO",
        "gate_count": len(rows),
        "open_count": len(rows),
        "closed_count": 0,
        "gates": rows,
        "next_action": "Independent review and evidence preparation only",
        "training_allowed": False,
        "model_inference_allowed": False,
        "energy_measured": False,
        "external_requests_allowed": False,
    }
