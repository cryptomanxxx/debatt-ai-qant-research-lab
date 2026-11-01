"""CL-5G: read-only H1 evidence boundary synthesis; no data or model compute."""
from __future__ import annotations

from integrations.continual_learning.cl3y_readiness_audit import scientific_readiness_audit

EVIDENCE = (
    ("CL5A", "generator source structure and nominal protocol constants",
     "STATIC_SOURCE_ONLY", "No generated records or pinned per-seed dataset hashes"),
    ("CL5B", "validation predicates for caller-supplied records",
     "VALIDATION_INTERFACE_ONLY", "No real dataset supplied or reproducibility established"),
    ("CL5C", "class-balanced XOR mean activity before uniqueness",
     "EXACT_ANALYTICAL_PROPOSAL", "Uniqueness-conditioned empirical activity remains unknown"),
    ("CL5D", "available XOR feature-space counts and necessary capacity",
     "EXACT_COMBINATORIAL_SUPPORT", "Finite rejection-sampler success not established"),
    ("CL5E", "class-conditioned independent-proposal collision probability",
     "EXACT_ANALYTICAL_PROXY", "Sequential globally deduplicated collision history not modelled"),
    ("CL5F", "uncapped expected draw-count lower floor for class quotas",
     "EXACT_ANALYTICAL_FLOOR", "Finite-attempt failures and actual cost remain unmeasured"),
)


def cl5g_evidence_boundary_audit() -> dict:
    """Check previous evidence against live NO_GO gates; do not close any gate."""
    readiness = scientific_readiness_audit()
    if (readiness["decision"] != "NO_GO" or readiness["open_count"] != 9
            or readiness["closed_count"] != 0 or readiness["training_allowed"] is not False
            or readiness["model_inference_allowed"] is not False):
        raise ValueError("unexpected change to scientific readiness state")
    if len(EVIDENCE) != 6 or [item[0] for item in EVIDENCE] != [
        "CL5A", "CL5B", "CL5C", "CL5D", "CL5E", "CL5F"
    ]:
        raise ValueError("evidence register incomplete or reordered")
    return {
        "kind": "CL5G_H1_EVIDENCE_BOUNDARY_AUDIT",
        "decision": "NO_GO",
        "evidence": [
            {"stage": stage, "supported_claim": claim, "evidence_class": evidence_class,
             "unresolved_limitation": limitation, "closes_scientific_gate": False}
            for stage, claim, evidence_class, limitation in EVIDENCE
        ],
        "scientific_gates_open": readiness["open_count"],
        "scientific_gates_closed": readiness["closed_count"],
        "next_evidence_needed": [
            "independent generator source/version and dataset-digest approval",
            "reviewed, separately authorized empirical data-validation protocol",
            "independent sparse/dense outcome-equivalence and instrumentation review",
            "preregistered statistical decision and calibrated energy boundary",
            "separate explicit authorization before any model experiment training",
        ],
        "dataset_generation_executed": False,
        "model_training_executed": False,
        "measured_energy": False,
        "scientific_approval_granted": False,
    }
