"""CL-8B four-arm CPU workload projection. NO model imports/data/training."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from integrations.continual_learning.cl6a_readiness_review import (
    PATH as PROTOCOL_PATH, review as review_cl6a,
)
from integrations.continual_learning.cl7f_h1_decision_audit import audit as audit_cl7f


@dataclass(frozen=True)
class FourArmPreflight:
    kind: str
    decision: str
    experiment_executable: bool
    model_training_authorized: bool
    proposed_seeds: int
    proposed_arms: int
    proposed_tasks: int
    proposed_checkpoints: int
    proposed_updates_per_arm_seed: int
    maximum_proposed_updates_all_arms_seeds: int
    maximum_proposed_heldout_predictions_all_arms_seeds: int
    open_scientific_gates: int
    open_scientific_findings: int
    physical_energy: str
    execution_status: str


def calculate(protocol: dict, readiness: dict, decision: dict) -> FourArmPreflight:
    reviewed = review_cl6a(protocol)
    if (reviewed["decision"] != "NO_GO" or reviewed["closed_gates"] != 0
            or decision.get("decision") != "NO_GO"
            or decision.get("open_cl6a_gates") != 9
            or decision.get("open_cl7a_findings") != 6):
        raise ValueError("scientific readiness has not been independently cleared")
    if (not isinstance(protocol, dict)
            or protocol.get("model_training_authorized") is not False
            or protocol.get("model_inference_authorized") is not False
            or protocol.get("dataset_generation_authorized") is not False):
        raise ValueError("unexpected execution authorization in draft")
    design = protocol["future_design"]
    seeds, arms, tasks, checkpoints = (
        len(design["seeds"]), len(protocol["arms"]),
        len(design["tasks"]), len(design["checkpoints"]),
    )
    if (seeds != 5 or arms != 4 or tasks != 4 or checkpoints != tasks + 1
            or design["maximum_updates_per_arm_seed"] !=
            design["train_records_per_task"] * tasks * design["passes_per_task"]
            or design["test_records_per_task"] != 64
            or protocol["model"]["shape"] != [16, 8, 2]
            or protocol["measurement_scope"]["hardware_energy"] != "NOT_MEASURED"):
        raise ValueError("proposed experiment scope drift; review a new preregistration")
    per_arm_seed = design["maximum_updates_per_arm_seed"]
    return FourArmPreflight(
        kind="CL8B_FOUR_ARM_PROJECTION_NOT_EXECUTED",
        decision="NO_GO",
        experiment_executable=False,
        model_training_authorized=False,
        proposed_seeds=seeds,
        proposed_arms=arms,
        proposed_tasks=tasks,
        proposed_checkpoints=checkpoints,
        proposed_updates_per_arm_seed=per_arm_seed,
        maximum_proposed_updates_all_arms_seeds=seeds * arms * per_arm_seed,
        maximum_proposed_heldout_predictions_all_arms_seeds=(
            seeds * arms * checkpoints * tasks * design["test_records_per_task"]
        ),
        open_scientific_gates=len(reviewed["open_gates"]),
        open_scientific_findings=decision["open_cl7a_findings"],
        physical_energy="NOT_MEASURED",
        execution_status="NEVER_EXECUTED_BY_THIS_PREFLIGHT",
    )


def preflight() -> FourArmPreflight:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    parent = audit_cl7f()
    return calculate(protocol, parent, parent)


def format_summary(item: FourArmPreflight) -> str:
    if item.decision != "NO_GO" or item.experiment_executable:
        raise ValueError("cannot present experiment as ready")
    return (
        "## CL-8B: Four-arm workload preflight (NO TRAINING)\n\n"
        "**Decision: NO_GO. All numbers below are proposed ceilings, not executed work.**\n\n"
        "| Proposed quantity | Count |\n| --- | ---: |\n"
        f"| Paired seeds | {item.proposed_seeds} |\n"
        f"| Architectures | {item.proposed_arms} |\n"
        f"| Tasks / checkpoints | {item.proposed_tasks} / {item.proposed_checkpoints} |\n"
        f"| Max updates per arm/seed | {item.proposed_updates_per_arm_seed:,} |\n"
        f"| Max proposed total updates | {item.maximum_proposed_updates_all_arms_seeds:,} |\n"
        f"| Max proposed heldout predictions | {item.maximum_proposed_heldout_predictions_all_arms_seeds:,} |\n"
        f"| Open CL-6A gates | {item.open_scientific_gates} |\n"
        f"| Open CL-7A findings | {item.open_scientific_findings} |\n\n"
        "**No training, inference, data generation, latency, DRAM traffic or physical energy was measured.**\n"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CL-8B static workload preflight")
    parser.add_argument("--preflight", action="store_true", required=True)
    parser.add_argument("--summary", type=Path, default=None)
    args = parser.parse_args(argv)
    result = preflight()
    print(json.dumps(asdict(result), indent=2, sort_keys=True))
    if args.summary is not None:
        # Only an explicitly supplied GitHub Actions summary path; never a dataset.
        args.summary.write_text(format_summary(result), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
