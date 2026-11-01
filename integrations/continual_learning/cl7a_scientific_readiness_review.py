"""CL-7A read-only critical protocol validation, not independent peer review."""
from __future__ import annotations

import json
from pathlib import Path

from integrations.continual_learning.cl6a_readiness_review import PATH as CL6A, review


PATH = Path(__file__).with_name("cl7a_scientific_design_critique.json")
EXPECTED_FINDINGS = (
    "construct_validity",
    "generator_selection_bias",
    "learnability_and_baselines",
    "causal_pairing",
    "measurement_construct",
    "statistics_and_reproducibility",
)


def validate_critique(critique: dict, parent: dict) -> dict:
    parent_status = review(parent)
    if parent_status["decision"] != "NO_GO" or parent_status["closed_gates"] != 0:
        raise ValueError("CL-6A must remain NO_GO with all nine gates open")
    if (not isinstance(critique, dict)
            or critique.get("kind") != "CL7A_SCIENTIFIC_DESIGN_CRITIQUE_NOT_INDEPENDENT_REVIEW"
            or critique.get("version") != "cl7a-v1"
            or critique.get("decision") != "NO_GO"
            or critique.get("open_cl6a_gates_expected") != len(parent_status["open_gates"])):
        raise ValueError("incorrect scientific review context or decision")
    if critique.get("analysis_only") is not True:
        raise ValueError("review preparation must be analysis-only")
    for flag in (
        "independent_scientific_signoff", "scientific_go_authorized",
        "model_construction_authorized", "model_inference_authorized",
        "training_authorized", "dataset_generation_authorized",
        "new_experiment_results", "energy_measured",
    ):
        if critique.get(flag) is not False:
            raise ValueError(f"unapproved scientific or execution claim: {flag}")
    findings = critique.get("findings")
    if not isinstance(findings, list) or [f.get("id") for f in findings] != list(EXPECTED_FINDINGS):
        raise ValueError("exact set of six methodological findings required")
    for finding in findings:
        if (finding.get("severity") != "blocking"
                or finding.get("status") != "OPEN"
                or not isinstance(finding.get("issue"), str)
                or len(finding["issue"].strip()) < 20
                or not isinstance(finding.get("required_evidence"), str)
                or len(finding["required_evidence"].strip()) < 20):
            raise ValueError("unresolved scientific finding missing evidence requirements")
    if (critique.get("parent_protocol") != CL6A.name
            or not isinstance(critique.get("scope"), str)
            or "H1" not in critique["scope"]
            or not isinstance(critique.get("conclusion"), str)
            or not critique["conclusion"].startswith("NOT_READY_FOR_TRAINING")):
        raise ValueError("unreviewed scope or readiness claim")
    return {
        "kind": "CL7A_CRITICAL_SCIENTIFIC_READINESS_REVIEW",
        "decision": "NO_GO",
        "open_scientific_findings": len(findings),
        "open_cl6a_gates": len(parent_status["open_gates"]),
        "independent_scientific_signoff": False,
        "new_experiment_results": False,
        "training_executed": False,
        "hardware_energy_measured": False,
    }


def audit() -> dict:
    return validate_critique(
        json.loads(PATH.read_text(encoding="utf-8")),
        json.loads(CL6A.read_text(encoding="utf-8")),
    )


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2, sort_keys=True))
