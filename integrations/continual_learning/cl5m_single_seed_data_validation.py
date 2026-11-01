"""CL-5M manually gated, single-seed synthetic DATA-ONLY validation.

Importing this module does not run generators. A command-line invocation must
provide both --seed 101 and --authorize-data-only. No model is trained.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from integrations.continual_learning.cl2_shared_rule_candidate import (
    generate_shared_rule_candidate,
    shared_rule_dataset_sha256,
)
from integrations.continual_learning.cl5b_supplied_data_validation import (
    compare_supplied_replicates,
    validate_supplied_splits,
)
from integrations.continual_learning.cl5l_instrumented_candidate import (
    generate_shared_rule_candidate_with_counters,
)

ROOT = Path(__file__).resolve().parent
SEED = 101


def validate_one_seed(*, seed: int, authorized: bool) -> dict:
    """Execute only after an explicit data-only command-line authorization.

    This function must never be called from standard test/CI workflows.
    """
    if type(seed) is not int or seed != SEED or authorized is not True:
        raise PermissionError("CL-5M requires explicit data-only authorization for seed 101")
    original = generate_shared_rule_candidate(seed)
    instrumented, reports = generate_shared_rule_candidate_with_counters(seed)
    original_digest = shared_rule_dataset_sha256(original)
    instrumented_digest = shared_rule_dataset_sha256(instrumented)
    comparison = compare_supplied_replicates(original, instrumented)
    if original_digest != instrumented_digest or comparison["matching"] is not True:
        raise ValueError("original and instrumented dataset digests differ")
    checked = validate_supplied_splits(original)
    if len(reports) != 8 or checked["records_checked"] != 768:
        raise ValueError("unexpected number of partitions or unique records")
    for report in reports:
        if (report["kind"] != "CL5L_INSTRUMENTED_GENERATOR_PARTITION"
                or report["generator_connected"] is not True
                or report["dataset_generation_executed"] is not True
                or report["attempt_cap_exhausted"] is not False
                or report["candidate_attempts"] != (
                    report["class_full_rejections"]
                    + report["duplicate_rejections_after_class_check"]
                    + report["accepted_records"]
                )):
            raise ValueError("instrumented counter provenance or conservation failure")
    sources = ("cl2_shared_rule_candidate.py", "cl5l_instrumented_candidate.py")
    return {
        "kind": "CL5M_ACTUAL_SINGLE_SEED_DATA_ONLY_OBSERVATION",
        "decision": "NO_GO",
        "authorized_scope": "synthetic data validation only",
        "seed": seed,
        "generator_source_sha256_observed": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in sources
        },
        "original_dataset_sha256": original_digest,
        "instrumented_dataset_sha256": instrumented_digest,
        "record_digests_match": True,
        "partitions": checked["partitions"],
        "instrumented_counters": list(reports),
        "records_checked": checked["records_checked"],
        "dataset_generation_executed": True,
        "model_training_executed": False,
        "model_inference_executed": False,
        "hardware_energy_measured": False,
        "scientific_gates_closed": 0,
        "scientific_approval_granted": False,
    }


def render_summary(result: dict) -> str:
    lines = [
        "## CL-5M: single-seed synthetic data-only validation",
        "",
        "> **DATA GENERATION ONLY — NO MODEL TRAINING OR ENERGY MEASUREMENT.** "
        "H1 remains NO_GO; no scientific gate is closed.",
        "",
        f"- Seed: `{result['seed']}`",
        f"- Unique records checked: `{result['records_checked']}`",
        f"- Original/instrumented digests match: `{result['record_digests_match']}`",
        f"- Dataset SHA-256: `{result['original_dataset_sha256']}`",
        "",
        "| Task | Split | Observed active fraction | Candidate attempts | Class-full rejects | Duplicate rejects | Accepted |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row, report in zip(result["partitions"], result["instrumented_counters"]):
        if row["task_id"] != report["task_id"] or row["partition"] != report["partition"]:
            raise ValueError("partition order differs between validation and counters")
        lines.append(
            f"| {row['task_id']} | {row['partition']} | "
            f"{row['observed_activity_rate']:.4f} | {report['candidate_attempts']} | "
            f"{report['class_full_rejections']} | "
            f"{report['duplicate_rejections_after_class_check']} | "
            f"{report['accepted_records']} |"
        )
    lines.extend(["", "These are software generator observations, not AI-learning or physical-energy results.", ""])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--authorize-data-only", action="store_true", required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args(argv)
    result = validate_one_seed(seed=args.seed, authorized=args.authorize_data_only)
    args.report.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.summary is not None:
        with args.summary.open("a", encoding="utf-8") as handle:
            handle.write(render_summary(result))
    print(render_summary(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
