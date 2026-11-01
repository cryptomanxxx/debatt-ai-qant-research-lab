"""CL-9I static, fail-closed H1 analysis contract checks. No model imports or runs."""
from __future__ import annotations

import json
from pathlib import Path

BASE = Path(__file__).resolve().parent
SEEDS = [101, 102, 103, 104, 105]
EVENTS = [
    "logical_parameter_reads", "logical_parameter_writes", "input_visits",
    "active_index_visits", "control_events", "nonlinear_events",
]
CONTRASTS = [
    ("primary_sparse", "local_hybrid", "sparse_local_hybrid",
     ["train", "heldout_inference"]),
    ("secondary_index_reuse", "sparse_local_hybrid", "sparse_shared_index",
     ["train"]),
]
CONTROL = "NO_INTERVENTION_CONTROL_STANDALONE_PREDICT_SAME_INHERITED_SPARSE_PATH"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(f"CL-9I NO_GO: {message}")


def load(path: Path) -> dict:
    obj = json.loads(path.read_text(encoding="utf-8"))
    require(type(obj) is dict, "contract must be an object")
    return obj


def validate(contract: dict, acceptance: dict, readiness: dict) -> dict:
    """Pure validation of preregistered method; never authorizes execution."""
    require(contract.get("kind") == "CL9H_PAIRED_H1_ANALYSIS_METHOD_REVIEW_ONLY"
            and contract.get("version") == "cl9h-v1", "unexpected contract identity")
    require(contract.get("decision") == "NO_GO"
            and contract.get("review_only") is True, "review-only decision changed")
    for field in ("independent_scientific_signoff", "model_construction_authorized",
                  "model_inference_authorized", "training_authorized",
                  "synthetic_data_generation_authorized", "h2_tested"):
        require(contract.get(field) is False, f"{field} must be false")
    require(type(contract.get("observations_recorded")) is int
            and contract["observations_recorded"] == 0, "observations present")
    require(contract.get("physical_energy") == "NOT_MEASURED", "energy claim changed")
    # Upstream safety cannot be delegated to a different workflow.
    require(readiness.get("kind") == "CL6A_CONTINUAL_LEARNING_READINESS_DRAFT"
            and readiness.get("version") == "cl6a-v1"
            and acceptance.get("kind") == "CL9B_H1_EXPERIMENT_ACCEPTANCE_CONTRACT"
            and acceptance.get("version") == "cl9b-v1", "upstream contract identity changed")
    for name, upstream, fields in (
        ("CL-6A", readiness, (
            "scientific_approved", "model_training_authorized",
            "dataset_generation_authorized", "model_inference_authorized",
            "physical_energy_measurement_authorized", "new_results_present")),
        ("CL-9B", acceptance, (
            "model_training_authorized", "model_inference_authorized",
            "dataset_generation_authorized", "energy_measured")),
    ):
        require(upstream.get("decision") == "NO_GO", f"{name} is not NO_GO")
        for field in fields:
            require(upstream.get(field) is False, f"{name}.{field} must be false")
    require(acceptance.get("review_only") is True
            and acceptance.get("status") == "PROPOSED_UNREVIEWED_THRESHOLDS",
            "CL-9B review status changed")
    require(readiness.get("execution_contract") == {
        "workflow_dispatch_training": False, "training_module_import": False,
        "auto_gate_closure": False, "training_go_live": False,
    }, "CL-6A execution fence changed")

    require(contract.get("planned_seeds") == SEEDS
            and acceptance.get("descriptive_seeds") == SEEDS
            and readiness.get("future_design", {}).get("seeds") == SEEDS,
            "seed agreement changed")
    require(contract.get("context_only_arm") == "dense_sgd", "context arm changed")
    require(contract.get("phases") == ["train", "heldout_inference"],
            "phase scope changed")
    require(contract.get("event_fields") == EVENTS
            and acceptance.get("work_metrics", {}).get("components") == EVENTS,
            "source event vector changed")

    contrasts = contract.get("contrast_definitions")
    require(type(contrasts) is list and len(contrasts) == 2,
            "contrasts missing or duplicated")
    for pair, expected in zip(contrasts, CONTRASTS):
        ident, reference, candidate, phases = expected
        require(type(pair) is dict and pair.get("id") == ident
                and pair.get("reference") == reference
                and pair.get("candidate") == candidate
                and pair.get("applicable_effect_phases") == phases,
                f"contrast/phase mismatch: {ident}")
        require(pair.get("causal_claim") ==
                ("sparse_traversal_only_under_same_local_rule" if ident == "primary_sparse"
                 else "active_index_reuse_only_under_same_sparse_local_rule"),
                f"causal claim changed: {ident}")
    require(contrasts[1].get("heldout_inference_role") == CONTROL,
            "secondary heldout is not an intervention")
    require(contract.get("outcomes_applicability") ==
            "Directional outcomes apply only to contrast-specific applicable_effect_phases; secondary heldout inference is a no-intervention control and has no index-reuse direction label",
            "outcome applicability relaxed")

    metric = contract.get("scalar_work_metric")
    require(type(metric) is dict
            and metric.get("status") == "UNAPPROVED_UNDEFINED"
            and "unit_cost_weights" in metric and metric["unit_cost_weights"] is None
            and "independent_approval_reference" in metric
            and metric["independent_approval_reference"] is None
            and metric.get("scalar_directional_claim_allowed") is False,
            "unapproved scalar metric enabled")
    require(contract.get("paired_delta") == "candidate_minus_reference"
            and contract.get("relative_delta") ==
            "signed_delta_divided_by_positive_reference_or_null_if_reference_zero",
            "delta or zero-denominator semantics changed")

    quality = contract.get("quality")
    source_quality = readiness.get("preregistered_quality", {})
    acceptance_quality = acceptance.get("quality_gate", {})
    require(type(quality) is dict
            and quality.get("final_mean_accuracy_floor_per_seed_and_arm") == 0.65
            and quality.get("maximum_absolute_final_mean_paired_accuracy_gap_per_seed") == 0.05
            and source_quality.get("final_mean_accuracy_floor") == 0.65
            and source_quality.get("paired_accuracy_difference_max_abs") == 0.05
            and acceptance_quality.get("final_mean_accuracy_floor_per_seed_and_arm") == 0.65
            and acceptance_quality.get("maximum_absolute_final_mean_paired_accuracy_gap_per_seed") == 0.05,
            "quality thresholds drifted")
    endpoint = quality.get("final_mean_accuracy_endpoint")
    require(type(endpoint) is dict and endpoint == {
        "source": "cl3d_analysis_contract.summarize_accuracy",
        "checkpoint": "post_T4", "checkpoint_row_index": -1,
        "heldout_tasks": ["T1", "T2", "T3", "T4"],
        "all_tasks_evaluated_at_every_checkpoint": True,
        "aggregation": "unweighted_mean_of_four_per_task_heldout_accuracies_at_post_T4",
        "checkpoint_pooling_allowed": False,
        "heldout_checkpoint_selection_allowed": False,
    }, "final heldout endpoint changed")
    require(quality.get("train_only_learnability_status") == "UNRESOLVED_REVIEW_REQUIRED"
            and quality.get("numerical_tolerance_status") == "UNRESOLVED_REVIEW_REQUIRED",
            "unreviewed quality gate treated as resolved")
    require(contract.get("outcomes") == {
        "DESCRIPTIVE_H1_SUPPORT": "all prerequisites pass and every paired seed delta < 0",
        "NO_OBSERVED_H1_ADVANTAGE": "all prerequisites pass and every paired seed delta >= 0",
        "INCONCLUSIVE": "any prerequisite fails or paired signs mixed",
    }, "directional decision rules changed")
    require(contract.get("minimum_common_prerequisites") == [
        "all_five_paired_seeds_complete", "source_and_per_seed_data_digests",
        "train_only_learnability_pass", "final_quality_and_parity_pass",
        "numeric_equivalence_pass", "identical_work_scope",
        "independent_preapproved_scalar_work_metric", "no_missing_or_excluded_attempts",
    ], "directional prerequisites changed")
    # Freeze both positive and negative rules *and* inconclusive precedence.
    common = ("All five planned paired seeds have complete frozen provenance; "
              "pass independently validated train-only learnability, final quality "
              "and numerical equivalence gates; and use observed source-event work "
              "under a fixed independently approved unit-cost metric with no missing "
              "or excluded runs.")
    require(acceptance.get("shared_directional_prerequisites") == common,
            "CL-9B shared prerequisites relaxed")
    require(acceptance.get("pilot_decision_rules") == [
        {
            "outcome": "DESCRIPTIVE_H1_SUPPORT",
            "rule": common + " Under these prerequisites, candidate work is strictly lower than reference in every paired seed.",
            "interpretation": "Narrow pilot support for source-level work reduction only.",
        },
        {
            "outcome": "NO_OBSERVED_H1_ADVANTAGE",
            "rule": common + " Under these same prerequisites, candidate work is greater than or equal to reference in every paired seed.",
            "interpretation": "No advantage in the tested tiny workload; not universal falsification.",
        },
        {
            "outcome": "INCONCLUSIVE",
            "rule": "Any failed common prerequisite (including absent provenance, learnability, unapproved score or failed seed), or mixed per-seed work signs. This outcome takes precedence whenever a common prerequisite is not satisfied.",
            "interpretation": "Do not declare a quality-matched saving. Report all available observations and failures.",
        },
    ], "CL-9B outcome semantics or inconclusive precedence changed")
    report = contract.get("reporting")
    require(type(report) is dict and report.get("no_posthoc_exclusions") is True
            and report.get("keep_all_attempts") is True
            and "undefined_relative_or_rate" in report
            and report["undefined_relative_or_rate"] is None
            and all(report.get(k) is True for k in (
                "report_phase_separately", "report_raw_unweighted_events",
                "no_generalizable_p_values", "no_energy_or_h2_claim")),
            "reporting safety changed")
    require(contract.get("unresolved_gates") == [
        "frozen_scope_and_weights", "independent_numeric_and_calibration_review",
        "verified_no_network_host", "independent_scientific_signoff",
        "marcus_separate_exact_execution_approval",
    ], "execution gates changed")
    return {"decision": "NO_GO", "validation": "STATIC_CONTRACT_ONLY",
            "planned_seeds": 5, "contrasts": 2, "observations": 0,
            "physical_energy": "NOT_MEASURED"}


def main() -> None:
    contract = load(BASE / "cl9h_paired_analysis_method.json")
    acceptance = load(BASE / "cl9b_h1_acceptance_contract.json")
    readiness = load(BASE / "cl6a_readiness_protocol.json")
    print(json.dumps(validate(contract, acceptance, readiness), sort_keys=True))


if __name__ == "__main__":
    main()
