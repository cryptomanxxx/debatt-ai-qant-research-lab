"""CL-7D static causal-identification protocol audit. No model import or run."""
from __future__ import annotations

import json
from pathlib import Path

from integrations.continual_learning.cl6a_readiness_review import PATH as CL6A, review
from integrations.continual_learning.cl7a_scientific_readiness_review import (
    PATH as CL7A, validate_critique,
)
from integrations.continual_learning.cl7b_h1_baseline_audit import (
    PATH as CL7B, validate_contract,
)
from integrations.continual_learning.cl7c_train_only_calibration_audit import (
    PATH as CL7C, validate_protocol,
)

PATH = Path(__file__).with_name("cl7d_causal_pairing_invariants.json")
UNVERIFIED = "NOT_VERIFIED"
PAIR_IDS = ("primary_sparse", "secondary_index_reuse")
EXPECTED_PAIRS = [
    {"id": "primary_sparse", "left": "local_hybrid",
     "right": "sparse_local_hybrid",
     "intervention": "first_layer_sparse_traversal_only", "status": UNVERIFIED},
    {"id": "secondary_index_reuse", "left": "sparse_local_hybrid",
     "right": "sparse_shared_index",
     "intervention": "active_index_reuse_only", "status": UNVERIFIED},
]
EXPECTED_REQUIREMENTS = {
    **{field: UNVERIFIED for field in (
        "initial_state_identical", "same_training_record_order",
        "same_task_order", "same_randomness_and_update_budget",
        "forward_outputs_equal_within_tolerance",
        "per_update_state_equal_within_tolerance",
        "active_feature_indices_equivalent",
        "instrumentation_same_event_scope")},
    "numerical_tolerance": "UNSPECIFIED_BLOCKING",
    "comparison_dtype_and_device": "UNSPECIFIED_BLOCKING",
}
EXPECTED_FAILURE = {
    "divergent_outputs_or_states": "STOP_CAUSAL_ATTRIBUTION",
    "missing_seed_or_trace": "REPORT_AND_STOP_CAUSAL_ATTRIBUTION",
    "changing_tolerance_after_observation": "FORBIDDEN",
    "changing_seed_order_after_observation": "FORBIDDEN",
    "logical_work_as_energy": "FORBIDDEN",
}
EXPECTED_BOUNDARIES = {
    "quality_matched_cost_claim_permitted": False,
    "catastrophic_forgetting_claim_permitted": False,
    "hardware_energy_claim_permitted": False,
    "all_parent_gates_open": True,
    "independent_validation_required": True,
    "separate_marcus_training_approval_required": True,
}
EXPECTED_FIELDS = {
    "kind", "version", "decision", "analysis_only", "parent_protocol",
    "critique", "baseline_contract", "calibration_protocol",
    "model_construction_authorized", "model_inference_authorized",
    "training_authorized", "dataset_generation_authorized",
    "independent_scientific_signoff", "new_results_present",
    "energy_measured", "proposed_paired_seeds", "causal_pairs",
    "context_only", "future_pairing_requirements", "failure_policy",
    "scientific_boundaries", "next_decision",
}


def validate_contract(
    contract: dict, parent: dict, critique: dict, baseline: dict, calibration: dict
) -> dict:
    parent_status = review(parent)
    critique_status = validate_critique(critique, parent)
    baseline_status = validate_contract_baseline(baseline, parent, critique)
    calibration_status = validate_protocol(calibration, parent, critique, baseline)
    if (parent_status["decision"] != "NO_GO"
            or parent_status["closed_gates"] != 0
            or critique_status["open_scientific_findings"] != 6
            or baseline_status["decision"] != "NO_GO"
            or calibration_status["decision"] != "NO_GO"):
        raise ValueError("prerequisite scientific NO_GO gates changed")
    if not isinstance(contract, dict) or set(contract) != EXPECTED_FIELDS:
        raise ValueError("unknown or missing CL-7D fields")
    fixed = {
        "kind": "CL7D_H1_CAUSAL_PAIRING_INVARIANTS_NOT_EXECUTED",
        "version": "cl7d-v1", "decision": "NO_GO",
        "analysis_only": True, "parent_protocol": CL6A.name,
        "critique": CL7A.name, "baseline_contract": CL7B.name,
        "calibration_protocol": CL7C.name,
        "proposed_paired_seeds": parent["future_design"]["seeds"],
        "causal_pairs": EXPECTED_PAIRS,
        "context_only": {
            "left": "dense_sgd", "right": "local_hybrid",
            "reason": "learning rule differs; not evidence for a sparsity-only intervention",
        },
        "future_pairing_requirements": EXPECTED_REQUIREMENTS,
        "failure_policy": EXPECTED_FAILURE,
        "scientific_boundaries": EXPECTED_BOUNDARIES,
        "next_decision": "Independently review numeric tolerance and paired state/trace observability before any separate authorization for model execution.",
    }
    for name, value in fixed.items():
        if contract.get(name) != value:
            raise ValueError(f"unexpected scientific readiness or causal contrast: {name}")
    for name in (
        "model_construction_authorized", "model_inference_authorized",
        "training_authorized", "dataset_generation_authorized",
        "independent_scientific_signoff", "new_results_present",
        "energy_measured",
    ):
        if contract.get(name) is not False:
            raise ValueError(f"unapproved model execution or result declaration: {name}")
    if [(x["id"], x["left"], x["right"]) for x in EXPECTED_PAIRS] != [
        (x["id"], x["left"], x["right"])
        for x in parent["comparisons"][:2]
    ]:
        raise ValueError("causal pair drift from CL-6A")
    return {
        "kind": "CL7D_STATIC_CAUSAL_PAIRING_AUDIT",
        "decision": "NO_GO",
        "causal_pairs_defined": len(PAIR_IDS),
        "numerical_equivalence_verified": False,
        "calibration_executed": False,
        "parent_gates_open": len(parent_status["open_gates"]),
        "open_scientific_findings": critique_status["open_scientific_findings"],
        "training_executed": False,
        "energy_measured": False,
    }


def validate_contract_baseline(baseline: dict, parent: dict, critique: dict) -> dict:
    return validate_contract_from_cl7b(baseline, parent, critique)


from integrations.continual_learning.cl7b_h1_baseline_audit import (
    validate_contract as validate_contract_from_cl7b,
)


def audit() -> dict:
    return validate_contract(
        json.loads(PATH.read_text(encoding="utf-8")),
        json.loads(CL6A.read_text(encoding="utf-8")),
        json.loads(CL7A.read_text(encoding="utf-8")),
        json.loads(CL7B.read_text(encoding="utf-8")),
        json.loads(CL7C.read_text(encoding="utf-8")),
    )


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2, sort_keys=True))
