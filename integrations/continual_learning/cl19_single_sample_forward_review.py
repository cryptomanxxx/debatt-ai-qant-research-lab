"""CL-19 first single-sample numerical comparison PROPOSAL ONLY.

No model imports or construction, inference, generator, training or energy.
The only CLI performs a read-only prerequisite audit and prints NO_GO.
"""
from __future__ import annotations

import json
from pathlib import Path

from cl16_pinned_dataset_audit import audit_repository_archive

HERE = Path(__file__).resolve().parent
SPEC = HERE / "cl19_single_sample_forward_candidate.json"
EXPECTED_ARMS = ["local_hybrid", "sparse_local_hybrid", "sparse_shared_index"]
EXPECTED_FEATURES = [0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]


class ExecutionNotAuthorized(RuntimeError):
    pass


def validate_review_only(spec: dict) -> dict:
    if (type(spec) is not dict
            or spec.get("kind") != "CL19_SINGLE_SAMPLE_FORWARD_NUMERIC_PROPOSAL"
            or spec.get("decision") != "NO_GO"
            or spec.get("review_only") is not True
            or spec.get("observations_recorded") != 0
            or any(spec.get(key) is not False for key in (
                "dataset_generation_authorized", "inference_authorized",
                "training_authorized", "energy_measurement_authorized",
                "execution_workflow_created"))):
        raise ValueError("CL-19 NO_GO: scope and execution status drift")
    audited = audit_repository_archive()
    source = spec.get("data_source", {})
    data = json.loads((HERE / "frozen_data_candidate" / "cl13" / "seed-101.json").read_text(encoding="ascii"))
    sample = data[0]["train"][0]
    if (source.get("location") !=
            "integrations/continual_learning/frozen_data_candidate/cl13/seed-101.json"
            or source.get("seed") != 101 or source.get("task") != "T1"
            or source.get("partition") != "train"
            or type(source.get("zero_based_index")) is not int
            or source["zero_based_index"] != 0
            or source.get("features") != EXPECTED_FEATURES
            or source.get("label") != 0 or source.get("heldout_access") is not False
            or sample != [EXPECTED_FEATURES, 0]
            or source.get("seed_dataset_sha256") != audited["seed_data_sha256"]["101"]):
        raise ValueError("CL-19 NO_GO: archived sample identity drift")
    scope = spec.get("model_scope", {})
    if (scope.get("arms") != EXPECTED_ARMS or scope.get("architecture") != [16, 8, 2]
            or scope.get("parameters_per_model") != 154
            or type(scope.get("initialization_seed")) is not int
            or scope.get("initialization_seed") != 101
            or scope.get("model_instances_max") != 3
            or scope.get("forward_calls_max") != 3
            or any(scope.get(name) != 0 for name in (
                "weight_updates_max", "training_steps_max", "loss_gradient_calls_max"))
            or scope.get("model_inference_performed") is not False):
        raise ValueError("CL-19 NO_GO: model budget drift")
    if spec.get("comparisons") != [
        {"pair_id": "primary_sparse", "left": "local_hybrid", "right": "sparse_local_hybrid"},
        {"pair_id": "secondary_index_reuse", "left": "sparse_local_hybrid", "right": "sparse_shared_index"},
    ]:
        raise ValueError("CL-19 NO_GO: paired comparison drift")
    numeric = spec.get("numeric_candidate", {})
    if (numeric.get("absolute") != 1e-12 or numeric.get("relative") != 1e-10
            or numeric.get("independently_approved") is not False
            or numeric.get("actual_parity_verified") is not False
            or numeric.get("trace_fields") != [
                "hidden", "logits", "active_indices", "initial_weights"]
            or numeric.get("no_training_or_loss_required") is not True):
        raise ValueError("CL-19 NO_GO: numeric candidate drift")
    runtime = spec.get("future_runtime", {})
    if (runtime.get("container_model_image_sha256") is not None
            or runtime.get("runner_image_sha") is not None
            or runtime.get("network_policy") != "NETWORK_NONE_REQUIRED_NOT_MODEL_VERIFIED"
            or any(runtime.get(k) is not None for k in (
                "memory_limit", "cpu_limit", "timeout_minutes",
                "marcus_specific_inference_authorization_reference"))
            or runtime.get("paid_runners_allowed") is not False
            or runtime.get("output_artifacts_max_mib") != 1
            or any(runtime.get(k) is not False for k in (
                "host_boundary_security_approved", "immutable_artifact_provenance_approved",
                "scientific_signoff", "security_signoff"))
            or spec.get("physical_energy") != "NOT_MEASURED"):
        raise ValueError("CL-19 NO_GO: unapproved runtime gate drift")
    return {
        "decision": "NO_GO",
        "scope": "SINGLE_SAMPLE_FORWARD_REVIEW_ONLY",
        "models_imported": 0,
        "models_constructed": 0,
        "forward_calls_executed": 0,
        "weight_updates_executed": 0,
        "training_authorized": False,
        "inference_authorized": False,
        "physical_energy": "NOT_MEASURED",
        "remaining_blockers": [
            "independent_numerical_tolerance_approval",
            "exact_model_image_runner_resource_budget_and_network_isolation",
            "independent_scientific_and_security_signoff",
            "immutable_artifact_provenance",
            "marcus_separate_exact_inference_authorization",
        ],
    }


def attempt_model_execution(*_args, **_kwargs) -> None:
    raise ExecutionNotAuthorized("CL-19 NO_GO: model inference not authorized")


if __name__ == "__main__":
    print(json.dumps(validate_review_only(json.loads(SPEC.read_text(encoding="ascii"))),
                     sort_keys=True))
