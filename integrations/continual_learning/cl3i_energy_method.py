"""CL-3I future hardware-energy measurement protocol (review-only).

This module NEVER accesses power meters, reads energy counters, runs a model,
or converts software operation proxies into physical energy.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass

METHOD_VERSION = "cl3i-physical-energy-method-draft-v0"
REQUIRED_FIELDS = (
    "meter_identity_and_calibration",
    "measurement_boundary",
    "idle_baseline_method",
    "sampling_resolution_and_uncertainty",
    "repetitions_and_warmup",
    "training_inference_phase_separation",
    "data_conversion_and_io_inclusion",
    "wall_time_and_workload_size",
    "paired_hardware_environment",
    "source_protocol_and_dataset_digests",
)


@dataclass(frozen=True)
class EnergyMethodDraft:
    version: str = METHOD_VERSION
    status: str = "DRAFT_NO_MEASUREMENTS"
    planned_boundary: str = "WHOLE_SYSTEM_AT_WALL_PREFERRED"
    physical_energy_joules: None = None
    meter_selected: bool = False
    meter_calibrated: bool = False
    training_approved: bool = False
    independent_scientific_review_approved: bool = False
    human_compute_approval: bool = False


def validate_draft(method: EnergyMethodDraft) -> None:
    """Strictly prohibit smuggling observed-energy or approval claims."""
    if not isinstance(method, EnergyMethodDraft):
        raise ValueError("expected energy-method draft")
    if (
        method.version != METHOD_VERSION
        or method.status != "DRAFT_NO_MEASUREMENTS"
        or method.planned_boundary != "WHOLE_SYSTEM_AT_WALL_PREFERRED"
        or method.physical_energy_joules is not None
        or method.meter_selected is not False
        or method.meter_calibrated is not False
        or method.training_approved is not False
        or method.independent_scientific_review_approved is not False
        or method.human_compute_approval is not False
    ):
        raise ValueError("measurement draft cannot assert results or approvals")


def prepare_method_review() -> dict:
    method = EnergyMethodDraft()
    validate_draft(method)
    return {
        "method": asdict(method),
        "mandatory_future_protocol_fields": list(REQUIRED_FIELDS),
        "training_allowed": False,
        "measurement_allowed": False,
        "energy_benefit_claim_allowed": False,
        "missing": [
            "calibrated_meter_and_scope",
            "hardware_sampling_and_uncertainty_protocol",
            "independent_scientific_review",
            "separate_human_compute_authorization",
        ],
        "warning": "Logical memory or operation counters are not joules; idle subtraction and I/O/conversion overhead must be measured at a declared boundary.",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CL-3I energy-method metadata only")
    parser.add_argument("--review-method", action="store_true", required=True)
    args = parser.parse_args(argv)
    if args.review_method:
        print(json.dumps(prepare_method_review(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
