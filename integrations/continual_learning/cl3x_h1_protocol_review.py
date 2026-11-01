"""CL-3X review-only H1 preregistration. Never runs a model or dataset.

Fails closed against legacy CL-3A governance and the existing CL-2 pilot.
"""
from __future__ import annotations

import json
from pathlib import Path
from collections.abc import Mapping

from integrations.continual_learning.cl2_paired_run_plan import load_candidate

ROOT = Path(__file__).parent
PROTOCOL_PATH = ROOT / "cl3x_h1_preregistration.json"
READINESS_PATH = ROOT / "cl3a_readiness.json"

REQUIRED_GATES = (
    "source_hash_and_dataset_digest_pin",
    "sparse_dense_local_equivalence_review",
    "paired_statistical_plan_review",
    "event_counter_scope_and_overhead_review",
    "non_test_calibration_and_learnability_review",
    "energy_boundary_or_explicit_not_measured",
    "runner_source_and_artifact_schema_review",
    "independent_scientific_review",
    "separate_explicit_marcus_compute_authorization",
)


def validate_draft(protocol: Mapping, candidate: Mapping, readiness: Mapping) -> None:
    """Prevent accidental training authorization or unsupported H1 claims."""
    if (
        protocol.get("protocol_id") != "cl3x-h1-sparse-activity-pilot-v0"
        or protocol.get("status") != "draft_review_only_unexecuted"
        or protocol.get("decision") != "NO_GO"
        or any(protocol.get(flag) is not False for flag in (
            "scientific_review_approved", "compute_approved", "runner_enabled",
            "experiment_training_allowed", "network_allowed", "hardware_energy_claim_allowed"
        ))
        or protocol.get("execution_entrypoint") is not None
        or protocol.get("results") is not None
    ):
        raise ValueError("CL-3X must remain a non-executable NO_GO draft")
    if readiness.get("decision") != "NO_GO" or readiness.get("compute_approved") is not False:
        raise ValueError("legacy scientific readiness gate changed")
    if candidate.get("compute_approved") is not False or candidate.get("runner_enabled") is not False:
        raise ValueError("underlying plan unexpectedly authorized")
    study = protocol.get("study_design")
    if not isinstance(study, dict) or (
        study.get("dataset") != candidate["dataset"]["candidate"]
        or study.get("dataset_source_sha256") is not None
        or study.get("dataset_digests_pinned") is not False
        or study.get("task_order") != candidate["dataset"]["task_order"]
        or study.get("paired_seeds") != candidate["paired_seeds"]
        or study.get("train_per_task") != candidate["dataset"]["train_per_task"]
        or study.get("test_per_task") != candidate["dataset"]["test_per_task"]
        or study.get("split_policy") != candidate["dataset"]["split_policy"]
        or study.get("task_id_supplied_to_model") is not False
        or study.get("checkpoints") != candidate["evaluation"]["checkpoints"]
        or study.get("heldout_updates_allowed") is not False
        or study.get("updates_per_seed_per_arm_ceiling") != candidate["training"]["updates_per_seed_ceiling"]
    ):
        raise ValueError("CL-3X provenance or paired protocol drift")
    if (
        study.get("primary_comparison") != {
            "control": "local_hybrid",
            "candidate": "sparse_local_hybrid",
            "purpose": "isolate sparse forward versus dense forward under same local update rule",
            "implementation_equivalence_verified": False,
        }
        or study.get("activity_stratification", {}).get("planned_buckets") != ["0-3", "4-7", "8-11", "12-16"]
        or study.get("activity_stratification", {}).get("realized_distribution_required") is not True
    ):
        raise ValueError("causal sparsity comparison not independently reviewed")
    secondary = study.get("secondary_comparisons")
    if not isinstance(secondary, list) or [(r.get("control"), r.get("candidate")) for r in secondary] != [
        ("sparse_local_hybrid", "sparse_shared_index"), ("dense_sgd", "local_hybrid")
    ]:
        raise ValueError("unsupported secondary comparisons")
    measure = protocol.get("measurement_plan")
    if not isinstance(measure, dict) or (
        measure.get("energy_status") != "NOT_MEASURED_AND_NOT_APPROVED"
        or measure.get("instrumentation_source_sha256") is not None
        or measure.get("meter_calibrated") is not False
        or measure.get("whole_system_boundary_documented") is not False
        or measure.get("training_inference_separate") is not True
        or not {"index_reads", "index_writes", "nonlinearity_counts"}.issubset(measure.get("model_work", []))
    ):
        raise ValueError("unsupported measurement or energy claims")
    interpretation = protocol.get("interpretation")
    if not isinstance(interpretation, dict) or (
        interpretation.get("quality_floor_final_mean_accuracy") != 0.65
        or interpretation.get("paired_accuracy_tolerance_percentage_points") != 5
        or interpretation.get("stop_on_failed_quality_gate") is not True
        or interpretation.get("stop_on_failed_provenance_or_integrity") is not True
        or interpretation.get("paired_seeds_inference") != "DESCRIPTIVE_ONLY_NO_SIGNIFICANCE_CLAIM_FROM_5_SEEDS"
    ):
        raise ValueError("scientific interpretation gates absent")
    if protocol.get("missing_review_gates") != list(REQUIRED_GATES):
        raise ValueError("review gates must remain explicit and open")
    compute = protocol.get("compute")
    if not isinstance(compute, dict) or (
        compute.get("estimated_cpu_budget_only") is not True
        or compute.get("external_calls_max") != 0
        or compute.get("cloud_budget_eur") != 0
    ):
        raise ValueError("no compute authorization in preregistration")


def review_summary() -> dict:
    """Read JSON and compare to existing plan; no model, IO beyond local files."""
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    readiness = json.loads(READINESS_PATH.read_text(encoding="utf-8"))
    candidate = load_candidate()
    validate_draft(protocol, candidate, readiness)
    return {
        "protocol_id": protocol["protocol_id"],
        "decision": "NO_GO",
        "primary_comparison": protocol["study_design"]["primary_comparison"],
        "seed_count": len(protocol["study_design"]["paired_seeds"]),
        "task_count": len(protocol["study_design"]["task_order"]),
        "missing_review_gates": protocol["missing_review_gates"],
        "training_allowed": False,
        "runner_available": False,
        "physical_energy_claim_allowed": False,
    }
