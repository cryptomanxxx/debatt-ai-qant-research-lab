"""CL-7B static baseline design audit. No dataset, model, or inference import."""
from __future__ import annotations

import json
from pathlib import Path

from integrations.continual_learning.cl6a_readiness_review import PATH as CL6A, review
from integrations.continual_learning.cl7a_scientific_readiness_review import (
    PATH as CL7A, validate_critique,
)

PATH = Path(__file__).with_name("cl7b_h1_baseline_contract.json")
EXPECTED_BASELINES = ("constant_zero", "train_majority", "random_uniform_expectation")
FROZEN_DEFINITIONS = (
    "Always predict class 0; compute heldout performance only after explicit separate inference authorization.",
    "For each seed, choose a single class using pooled training labels only; ties choose class 0; freeze before heldout scoring.",
    "Binary balanced-label reference with expected accuracy 0.5; this is not a sampled classifier or an observed empirical result.",
)
FROZEN_FAILURE_POLICY = (
    "Report all proposed seeds and failed runs without posthoc exclusion; "
    "suspend quality-matched cost claims if any required paired seed is invalid."
)


def validate_contract(contract: dict, parent: dict, critique: dict) -> dict:
    cl6_status = review(parent)
    cl7_status = validate_critique(critique, parent)
    if cl6_status["decision"] != "NO_GO" or cl6_status["closed_gates"] != 0:
        raise ValueError("CL-6A must remain NO_GO with all gates open")
    if cl7_status["decision"] != "NO_GO" or cl7_status["open_scientific_findings"] != 6:
        raise ValueError("CL-7A scientific findings must remain open")
    if (not isinstance(contract, dict)
            or contract.get("kind") != "CL7B_H1_BASELINE_AND_QUALITY_DESIGN_NOT_EXECUTED"
            or contract.get("version") != "cl7b-v1"
            or contract.get("decision") != "NO_GO"
            or contract.get("parent") != CL6A.name
            or contract.get("scientific_critique") != CL7A.name
            or contract.get("open_cl6a_gates_expected") != 9
            or contract.get("open_cl7a_findings_expected") != 6):
        raise ValueError("invalid baseline document identity or readiness")
    if contract.get("analysis_only") is not True:
        raise ValueError("not analysis-only")
    for key in ("independent_scientific_signoff", "model_construction_authorized",
                "model_inference_authorized", "training_authorized",
                "dataset_generation_authorized", "new_results_present",
                "energy_measured"):
        if contract.get(key) is not False:
            raise ValueError("unapproved execution or evidence flag: " + key)
    baselines = contract.get("baseline_definitions")
    if not isinstance(baselines, list) or tuple(
        baseline.get("id") for baseline in baselines if isinstance(baseline, dict)
    ) != EXPECTED_BASELINES or len(baselines) != len(EXPECTED_BASELINES):
        raise ValueError("expected three distinct immutable baseline definitions")
    expected_types = ("nonlearning", "nonlearning", "analytic")
    expected_fit = ("none", "training_only", "none")
    for item, expected_type, expected_partition, frozen_definition in zip(
        baselines, expected_types, expected_fit, FROZEN_DEFINITIONS
    ):
        if (item.get("type") != expected_type
                or item.get("fit_partition") != expected_partition
                or item.get("test_label_tuning") is not False
                or item.get("definition") != frozen_definition):
            raise ValueError("baseline must forbid heldout selection and specify procedure")
    calibration = contract.get("calibration")
    if (not isinstance(calibration, dict)
            or calibration.get("status") != "UNPERFORMED"
            or calibration.get("test_labels_used") is not False
            or calibration.get("numerical_tolerance") != "UNAPPROVED"
            or "Training-only" not in calibration.get("scope", "")):
        raise ValueError("calibration not authorized or falsely claimed")
    quality = contract.get("quality_interpretation")
    prereg = parent["preregistered_quality"]
    if (not isinstance(quality, dict)
            or quality.get("final_mean_accuracy_floor") != prereg["final_mean_accuracy_floor"]
            or quality.get("paired_accuracy_difference_max_abs") != prereg["paired_accuracy_difference_max_abs"]
            or quality.get("per_seed_per_primary_and_secondary_contrast") is not True
            or quality.get("requires_baselines_and_calibration") is not True
            or quality.get("quality_matched_cost_claim_permitted_now") is not False
            or quality.get("accuracy_floor_alone_sufficient") is not False
            or quality.get("failure_or_missing_run_policy") != FROZEN_FAILURE_POLICY):
        raise ValueError("incorrect quality claim or preregistration")
    gates = contract.get("gates")
    if not isinstance(gates, dict) or set(gates) != {
        "learnability_and_baselines", "causal_pairing",
        "independent_scientific_review", "separate_marcus_training_authorization"
    } or any(value != "OPEN" for value in gates.values()):
        raise ValueError("all CL-7B scientific blockers must remain open")
    if ("H1" not in contract.get("scope", "")
            or "not catastrophic forgetting" not in contract["scope"]):
        raise ValueError("invalid scientific construct scope")
    return {
        "kind": "CL7B_STATIC_BASELINE_AUDIT",
        "decision": "NO_GO",
        "baselines_specified": len(baselines),
        "calibration_performed": False,
        "cl6a_open_gates": len(cl6_status["open_gates"]),
        "cl7a_open_findings": cl7_status["open_scientific_findings"],
        "training_executed": False,
        "results_present": False,
    }


def audit() -> dict:
    return validate_contract(
        json.loads(PATH.read_text(encoding="utf-8")),
        json.loads(CL6A.read_text(encoding="utf-8")),
        json.loads(CL7A.read_text(encoding="utf-8")),
    )


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2, sort_keys=True))
