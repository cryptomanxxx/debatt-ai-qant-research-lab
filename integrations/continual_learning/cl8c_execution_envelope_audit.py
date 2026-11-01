"""CL-8C future execution-envelope validator. NEVER runs a child process."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from integrations.continual_learning.cl8b_four_arm_preflight import preflight

PATH = Path(__file__).with_name("cl8c_cpu_execution_envelope.json")
FALSE_FLAGS = (
    "execution_entrypoint_available", "model_execution_authorized",
    "training_authorized", "inference_authorized",
    "dataset_generation_authorized", "network_isolation_implemented",
    "child_process_resource_enforcement_implemented",
    "immutable_artifact_writer_implemented", "kill_switch_implemented",
    "independent_scientific_signoff", "new_results_present",
)
EXPECTED_LIMITS = {
    "proposed_max_cpu_seconds_per_worker": 120,
    "proposed_max_wall_seconds_per_worker": 300,
    "proposed_max_address_space_bytes": 1073741824,
    "proposed_max_output_bytes_per_run": 20971520,
    "proposed_max_updates_per_arm_seed": 512,
    "proposed_max_arms": 4,
    "proposed_max_seeds": 5,
    "proposed_max_total_updates": 10240,
}
REQUIRED_IMPLEMENTATION = (
    "offline_no_network_child_process_enforcement",
    "linux_child_cpu_and_address_space_rlimits",
    "external_wall_clock_timeout_with_child_tree_termination",
    "output_disk_budget_and_artifact_hash_verification",
    "immutable_reviewed_commit_protocol_and_input_manifest",
    "independent_scientific_go_and_separate_marcus_training_approval",
)
FORBIDDEN_FEATURES = (
    "user_defined_command_execution",
    "shell_execution",
    "network_download",
    "automatic_training_workflow",
    "inline_approval_boolean_as_authority",
)
EXPECTED_FIELDS = {
    "kind", "version", "decision", "review_only", *FALSE_FLAGS,
    "parent_protocol", "parent_decision_matrix", "limits",
    "required_implementation_before_any_execution", "unapproved_features",
    "approval_claim", "energy",
}


def validate_envelope(document: dict, projection: object) -> dict:
    if not isinstance(document, dict) or set(document) != EXPECTED_FIELDS:
        raise ValueError("missing/unknown envelope field")
    fixed = {
        "kind": "CL8C_FUTURE_CPU_EXECUTION_ENVELOPE_NOT_IMPLEMENTED",
        "version": "cl8c-v1",
        "decision": "NO_GO",
        "review_only": True,
        "parent_protocol": "cl6a_readiness_protocol.json",
        "parent_decision_matrix": "cl7f_h1_decision_matrix.json",
        "limits": EXPECTED_LIMITS,
        "required_implementation_before_any_execution": list(REQUIRED_IMPLEMENTATION),
        "unapproved_features": list(FORBIDDEN_FEATURES),
        "approval_claim": (
            "This proposal defines ceilings only. No OS resource limits or "
            "network isolation are currently installed by this module."
        ),
        "energy": "NOT_MEASURED",
    }
    for name, value in fixed.items():
        if document.get(name) != value:
            raise ValueError(f"unapproved execution-envelope change: {name}")
    for name in FALSE_FLAGS:
        if document.get(name) is not False:
            raise ValueError(f"unapproved compute authority or unimplemented guard: {name}")
    if (
        getattr(projection, "decision", None) != "NO_GO"
        or getattr(projection, "experiment_executable", None) is not False
        or getattr(projection, "maximum_proposed_updates_all_arms_seeds", None)
        != EXPECTED_LIMITS["proposed_max_total_updates"]
        or getattr(projection, "proposed_updates_per_arm_seed", None)
        != EXPECTED_LIMITS["proposed_max_updates_per_arm_seed"]
        or getattr(projection, "proposed_arms", None)
        != EXPECTED_LIMITS["proposed_max_arms"]
        or getattr(projection, "proposed_seeds", None)
        != EXPECTED_LIMITS["proposed_max_seeds"]
    ):
        raise ValueError("CL-8B four-arm ceiling or readiness drift")
    return {
        "kind": "CL8C_STATIC_LIMIT_PROPOSAL",
        "decision": "NO_GO",
        "runtime_enforced": False,
        "network_isolated": False,
        "execution_entrypoint_available": False,
        "model_training_executed": False,
        "proposed_limits": dict(EXPECTED_LIMITS),
        "required_implementation_count": len(REQUIRED_IMPLEMENTATION),
    }


def audit() -> dict:
    return validate_envelope(
        json.loads(PATH.read_text(encoding="utf-8")),
        preflight(),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CL-8C review-only resource limit proposal")
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--validate", action="store_true")
    modes.add_argument("--execute", action="store_true", help="PROHIBITED")
    args = parser.parse_args(argv)
    if args.execute:
        parser.error(
            "No runner exists; no compute/network sandbox was implemented or approved. "
            "Model execution and training remain PROHIBITED."
        )
    print(json.dumps(audit(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
