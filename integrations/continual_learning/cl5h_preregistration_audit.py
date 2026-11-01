"""CL-5H read-only draft-contract audit; does not import or call generator."""
from __future__ import annotations

import json
from pathlib import Path

from integrations.continual_learning.cl3y_readiness_audit import scientific_readiness_audit

DRAFT_PATH = Path(__file__).with_name("cl5h_empirical_data_audit_draft.json")

REQUIRED_RECORD_FIELDS = (
    "generator_source_sha256",
    "canonical_dataset_sha256",
    "task_partition_counts",
    "xor_class_balance",
    "empirical_activity_by_task_and_partition",
    "within_seed_duplicate_count",
    "cross_partition_duplicate_count",
    "rejection_attempts_by_task_and_partition",
    "generator_attempt_cap_outcome",
    "validation_failures",
)


def cl5h_preregistration_audit() -> dict:
    draft = json.loads(DRAFT_PATH.read_text(encoding="utf-8"))
    readiness = scientific_readiness_audit()
    expected_design = {
        "seeds": [101, 102, 103, 104, 105],
        "task_order": ["T1", "T2", "T3", "T4"],
        "train_per_task": 128,
        "test_per_task": 64,
        "global_label": "features[0] XOR features[1]",
        "global_uniqueness_within_seed": True,
    }
    expected_authorization = {
        "data_generation_authorized": False,
        "experiment_training_authorized": False,
        "permission_to_run_in_CI": False,
    }
    if (draft["decision"] != "NO_GO" or draft["status"] != "DRAFT_NOT_AUTHORIZED"
            or draft["planned_design"] != expected_design
            or draft["authorization"] != expected_authorization
            or draft["generator"]["source_sha256_approved"] is not None
            or draft["required_future_fields_per_seed"] != list(REQUIRED_RECORD_FIELDS)
            or draft["scientific_claims"] != {
                "dataset_generated": False, "reproducibility_proven": False,
                "training_performed": False, "hardware_energy_measured": False
            }
            or draft["output_record_policy"] != {
                "immutable_per_seed_artifacts_required": True,
                "samples_may_be_published": False,
                "measure_hardware_energy": False
            }
            or readiness["decision"] != "NO_GO" or readiness["open_count"] != 9):
        raise ValueError("CL-5H approval or evidence boundary changed")
    return {
        "kind": "CL5H_PREREGISTRATION_ONLY",
        "decision": "NO_GO",
        "future_result_fields": list(REQUIRED_RECORD_FIELDS),
        "data_generation_authorized": False,
        "experiment_training_authorized": False,
        "scientific_gates_open": 9,
        "dataset_generation_executed": False,
        "model_training_executed": False,
    }
