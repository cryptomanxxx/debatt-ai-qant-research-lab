"""CL-3B offline preflight: NO model training or experiment execution path.

Only validates committed draft metadata and prints an estimate of the
maximum proposed bounded workload. --execute is deliberately prohibited.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from integrations.continual_learning.cl2_paired_run_plan import load_candidate
from integrations.continual_learning.cl3b_result_contract import validate_result_shape


READINESS_PATH = Path(__file__).with_name("cl3a_readiness.json")


@dataclass(frozen=True)
class Preflight:
    protocol_id: str
    decision: str
    execution_allowed: bool
    paired_seed_count: int
    model_arm_count: int
    task_count: int
    max_proposed_training_updates_total: int
    max_proposed_heldout_predictions_total: int
    reasons: tuple[str, ...]


def prepare_preflight() -> Preflight:
    """Read existing proposals; never instantiate or update a model."""
    plan = load_candidate()
    review = json.loads(READINESS_PATH.read_text(encoding="utf-8"))
    blockers = review.get("mandatory_blockers")
    if not isinstance(blockers, list) or not blockers:
        raise ValueError("readiness blockers missing")
    if (
        review.get("decision") != "NO_GO"
        or review.get("go_for_training") is not False
        or review.get("compute_approved") is not False
        or review.get("runner_enabled") is not False
        or any(not isinstance(b, dict) or b.get("closed") is not False for b in blockers)
    ):
        raise ValueError("preflight expects the locked NO_GO readiness draft")
    n_seeds = len(plan["paired_seeds"])
    n_arms = len(plan["models"])
    n_tasks = len(plan["dataset"]["task_order"])
    max_updates = (
        plan["training"]["updates_per_seed_ceiling"] * n_arms * n_seeds
    )
    n_checkpoints = len(plan["evaluation"]["checkpoints"])
    max_predictions = (
        n_seeds * n_arms * n_tasks
        * n_checkpoints * plan["dataset"]["test_per_task"]
    )
    # Verify checkpoint shape contract without creating any experimental rows.
    validate_result_shape(n_checkpoints, n_tasks)
    return Preflight(
        protocol_id=plan["protocol_id"],
        decision="NO_GO",
        execution_allowed=False,
        paired_seed_count=n_seeds,
        model_arm_count=n_arms,
        task_count=n_tasks,
        max_proposed_training_updates_total=max_updates,
        max_proposed_heldout_predictions_total=max_predictions,
        reasons=tuple(str(b["id"]) for b in blockers),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CL-3B safe no-compute preflight")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true", help="print preflight JSON; no training")
    mode.add_argument("--execute", action="store_true", help="INTENTIONALLY DISABLED")
    args = parser.parse_args(argv)
    if args.execute:
        parser.error(
            "experiment execution is NOT IMPLEMENTED OR AUTHORIZED; "
            "requires an independently reviewed future runner and separate compute approval"
        )
    report = prepare_preflight()
    print(json.dumps(asdict(report), sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
