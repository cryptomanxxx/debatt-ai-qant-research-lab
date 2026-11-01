"""CL-7F read-only H1 scientific readiness matrix; no model or data access."""
from __future__ import annotations

import json
from pathlib import Path

from integrations.continual_learning.cl6a_readiness_review import PATH as CL6A, review
from integrations.continual_learning.cl7a_scientific_readiness_review import (
    PATH as CL7A, validate_critique,
)
from integrations.continual_learning.cl7b_h1_baseline_audit import (
    PATH as CL7B, validate_contract as validate_baseline,
)
from integrations.continual_learning.cl7c_train_only_calibration_audit import (
    PATH as CL7C, validate_protocol as validate_calibration,
)
from integrations.continual_learning.cl7d_causal_pairing_audit import (
    PATH as CL7D, validate_contract as validate_pairing,
)
from integrations.continual_learning.cl7e_measurement_boundary_audit import (
    PATH as CL7E, validate_contract as validate_measurement,
)

PATH = Path(__file__).with_name("cl7f_h1_decision_matrix.json")
PREREQUISITES = (CL7A, CL7B, CL7C, CL7D, CL7E)
EXPECTED_EVIDENCE = (
    "Freeze source and per-seed sample digests; review leakage, activity and label selection using separately authorized data generation.",
    "Independently approve numeric tolerance, dtype/device, identical initialization and per-update paired parity.",
    "Review five-seed descriptive uncertainty, quality parity, train-only baselines and failed-run reporting before any observations.",
    "Approve observer-overhead comparison, repeat/warmup, CPU environment and training/inference event boundaries.",
    "Freeze no-leakage train/calibration split, control capacity, and success criterion before any calibration model run.",
    "Keep physical energy NOT_MEASURED for first CPU experiment unless independently reviewed meter boundary is approved.",
    "Independently review offline runner limits, exact commit, input manifests, kill switch, artifact hashes, and no-network safeguards.",
    "Obtain independent domain-scientist critique of H1 construct validity and causal interpretation; Codex is not such signoff.",
    "Request Marcus's separate informed permission for exact reviewed commit, dataset digest, CPU/time budget, and training scope.",
)
EXPECTED_SEQUENCE = [
    "Freeze unresolved train-only calibration and pairing acceptance criteria through scientific review.",
    "Review bounded offline runner, immutable inputs, and all evidence instrumentation.",
    "Resolve independent scientific review and all non-user gates or explicitly retain a NO_GO decision.",
    "Seek separate explicit Marcus authorization for a specified minimal CPU experiment only after prerequisites are met.",
]
EXPECTED_CLAIMS = [
    "measured_joules_per_learning_step",
    "hardware_photonic_or_loihi_speedup",
    "verified_quality_matched_computation_savings",
    "mitigated_catastrophic_forgetting",
]
EXPECTED_FIELDS = {
    "kind", "version", "decision", "review_only", "independent_scientific_signoff",
    "execution_authorized", "training_authorized", "model_inference_authorized",
    "dataset_generation_authorized", "new_results_present",
    "hardware_energy_measured", "parent", "prerequisites", "readiness_gates",
    "proposed_sequence", "claims_not_supported", "open_gates_expected",
    "open_scientific_findings_expected", "experiment_run_count_in_this_step",
}


def validate_matrix(document: dict, parent: dict, critique: dict,
                    baseline: dict, calibration: dict, pairing: dict,
                    measurement: dict) -> dict:
    p = review(parent)
    a = validate_critique(critique, parent)
    b = validate_baseline(baseline, parent, critique)
    c = validate_calibration(calibration, parent, critique, baseline)
    d = validate_pairing(pairing, parent, critique, baseline, calibration)
    e = validate_measurement(measurement, parent, critique, baseline, calibration, pairing)
    if (p["decision"] != "NO_GO" or p["closed_gates"] != 0
            or a["open_scientific_findings"] != 6
            or any(x["decision"] != "NO_GO" for x in (b, c, d, e))):
        raise ValueError("upstream gates do not preserve NO_GO")
    if not isinstance(document, dict) or set(document) != EXPECTED_FIELDS:
        raise ValueError("unknown or missing review-only matrix keys")
    fixed = {
        "kind": "CL7F_H1_DECISION_MATRIX_REVIEW_ONLY",
        "version": "cl7f-v1", "decision": "NO_GO", "review_only": True,
        "parent": CL6A.name,
        "prerequisites": [path.name for path in PREREQUISITES],
        "proposed_sequence": EXPECTED_SEQUENCE,
        "claims_not_supported": EXPECTED_CLAIMS,
        "open_gates_expected": 9, "open_scientific_findings_expected": 6,
        "experiment_run_count_in_this_step": 0,
    }
    for key, expected in fixed.items():
        if document.get(key) != expected:
            raise ValueError(f"unreviewed evidence claim or matrix drift: {key}")
    for key in ("independent_scientific_signoff", "execution_authorized",
                "training_authorized", "model_inference_authorized",
                "dataset_generation_authorized", "new_results_present",
                "hardware_energy_measured"):
        if document.get(key) is not False:
            raise ValueError(f"false research authorization or evidence: {key}")
    gates = document.get("readiness_gates")
    if not isinstance(gates, list) or len(gates) != len(EXPECTED_EVIDENCE):
        raise ValueError("expected nine unresolved prerequisite gates")
    for actual, gate_id, evidence in zip(
        gates, parent["open_gates"], EXPECTED_EVIDENCE
    ):
        if actual != {"id": gate_id, "status": "OPEN", "next_evidence": evidence}:
            raise ValueError(f"unreviewed readiness transition: {gate_id}")
    return {
        "kind": "CL7F_STATIC_H1_DECISION_REVIEW",
        "decision": "NO_GO",
        "open_cl6a_gates": len(p["open_gates"]),
        "open_cl7a_findings": a["open_scientific_findings"],
        "experiment_run_count": 0,
        "model_training_executed": False,
        "physical_energy_measured": False,
        "independent_scientific_signoff": False,
    }


def audit() -> dict:
    return validate_matrix(*(
        json.loads(path.read_text(encoding="utf-8"))
        for path in (PATH, CL6A, *PREREQUISITES)
    ))


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2, sort_keys=True))
