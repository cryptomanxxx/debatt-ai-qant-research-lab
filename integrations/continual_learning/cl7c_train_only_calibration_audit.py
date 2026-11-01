"""CL-7C source-only validation of a NOT_RUN calibration proposal."""
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

PATH = Path(__file__).with_name("cl7c_train_only_calibration_protocol.json")

FROZEN_SPLIT = {
    "status": "UNSPECIFIED_BLOCKING",
    "split_unit": "unique_feature_vector",
    "group_identical_vectors_together": True,
    "heldout_test_records_eligible": False,
    "heldout_test_labels_eligible": False,
    "per_seed_split_indices_frozen": False,
    "seed_specific_train_validation_subsets_frozen": False,
    "class_coverage_verified": False,
    "overlap_audited": False,
    "reason": "No source-approved sample-level records or independent calibration split protocol exists; CL-5M aggregate seed-101 data are insufficient.",
}
FROZEN_FENCES = {
    "heldout_labels_for_hyperparameters": False,
    "heldout_checkpoint_selection": False,
    "cross_seed_best_run_selection": False,
    "posthoc_threshold_adjustment": False,
    "reuse_calibration_validation_for_final_report": False,
    "freeze_all_future_rules_before_data_access": True,
}
FROZEN_RULES = {
    "accuracy_floor_source": "cl6a_readiness_protocol.json:preregistered_quality.final_mean_accuracy_floor",
    "paired_tolerance_source": "cl6a_readiness_protocol.json:preregistered_quality.paired_accuracy_difference_max_abs",
    "train_only_learnability_threshold": "UNSPECIFIED_BLOCKING",
    "control_model_specification": "UNSPECIFIED_BLOCKING",
    "numerical_equivalence_tolerance": "UNSPECIFIED_BLOCKING",
    "quality_matched_cost_claim_permitted": False,
    "failed_seed_handling": "REPORT_ALL_AND_SUSPEND_CLAIMS",
}
FROZEN_CONTROLS = [
    {"id": "train_majority",
     "reference": "cl7b_h1_baseline_contract.json:train_majority",
     "uses_labels_from": "training_only", "executed": False},
    {"id": "train_only_learned_control",
     "reference": "future separately reviewed capacity-matched learnability control",
     "uses_labels_from": "training_only", "executed": False},
]


def validate_protocol(protocol: dict, parent: dict, critique: dict, baseline: dict) -> dict:
    parent_result = review(parent)
    critique_result = validate_critique(critique, parent)
    baseline_result = validate_contract(baseline, parent, critique)
    if (parent_result["decision"] != "NO_GO"
            or parent_result["closed_gates"] != 0
            or critique_result["open_scientific_findings"] != 6
            or baseline_result["decision"] != "NO_GO"):
        raise ValueError("parent research gates must remain OPEN / NO_GO")
    if not isinstance(protocol, dict):
        raise ValueError("calibration proposal must be a mapping")
    expected_fields = {
        "kind", "version", "decision", "analysis_only", "parent_protocol",
        "baseline_contract", "critique", "model_construction_authorized",
        "model_inference_authorized", "training_authorized",
        "dataset_generation_authorized", "independent_scientific_signoff",
        "new_results_present", "energy_measured", "calibration_status",
        "fixed_study_seeds", "proposed_task_ids", "source_partition",
        "calibration_split", "future_control_definitions", "selection_fences",
        "acceptance_rules", "open_gates_preserved", "next_decision",
    }
    if set(protocol) != expected_fields:
        raise ValueError("unrecognized or missing calibration contract field")
    identity = {
        "kind": "CL7C_TRAIN_ONLY_LEARNABILITY_CALIBRATION_DESIGN_NOT_EXECUTED",
        "version": "cl7c-v1", "decision": "NO_GO", "analysis_only": True,
        "parent_protocol": CL6A.name, "baseline_contract": CL7B.name,
        "critique": CL7A.name, "calibration_status": "NOT_RUN",
        "source_partition": "training_records_only",
        "fixed_study_seeds": parent["future_design"]["seeds"],
        "proposed_task_ids": parent["future_design"]["tasks"],
        "open_gates_preserved": {"cl6a": 9, "cl7a": 6, "cl7b_baselines": 3},
        "next_decision": "Freeze and externally review split and control designs before seeking separate explicit data access, inference or training authorization.",
    }
    for key, expected in identity.items():
        if protocol.get(key) != expected:
            raise ValueError(f"invalid calibration identity or readiness: {key}")
    for key in ("model_construction_authorized", "model_inference_authorized",
                "training_authorized", "dataset_generation_authorized",
                "independent_scientific_signoff", "new_results_present",
                "energy_measured"):
        if protocol.get(key) is not False:
            raise ValueError(f"unapproved evidence or execution claim: {key}")
    if protocol.get("calibration_split") != FROZEN_SPLIT:
        raise ValueError("calibration split or heldout leakage fence altered")
    if protocol.get("selection_fences") != FROZEN_FENCES:
        raise ValueError("heldout selection fence altered")
    if protocol.get("acceptance_rules") != FROZEN_RULES:
        raise ValueError("future quality or run-failure interpretation altered")
    if protocol.get("future_control_definitions") != FROZEN_CONTROLS:
        raise ValueError("future control definitions altered")
    return {
        "kind": "CL7C_CALIBRATION_READINESS_AUDIT",
        "decision": "NO_GO",
        "calibration_status": "NOT_RUN",
        "split_defined": False,
        "controls_executed": False,
        "open_parent_gates": len(parent_result["open_gates"]),
        "open_scientific_findings": critique_result["open_scientific_findings"],
        "new_results_present": False,
        "training_executed": False,
    }


def audit() -> dict:
    return validate_protocol(
        json.loads(PATH.read_text(encoding="utf-8")),
        json.loads(CL6A.read_text(encoding="utf-8")),
        json.loads(CL7A.read_text(encoding="utf-8")),
        json.loads(CL7B.read_text(encoding="utf-8")),
    )


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2, sort_keys=True))
