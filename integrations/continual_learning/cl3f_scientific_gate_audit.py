"""CL-3F read-only scientific gate matrix; never authorizes training.

Evidence paths point to PREPARATORY code and documents, not completed
scientific reviews. Missing approvals remain missing even if files exist.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from integrations.continual_learning.cl2_paired_run_plan import load_candidate

HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parent.parent
READINESS = HERE / "cl3a_readiness.json"

# Candidate evidence is NOT proof that a scientific blocker is closed.
EVIDENCE = {
    "dataset_provenance": ("integrations/continual_learning/cl3c_dataset_audit.py",),
    "construct_validity": ("integrations/continual_learning/CL3C_DATASET_PROVENANCE.md",),
    "pilot_learnability": (),
    "comparability": (
        "integrations/continual_learning/cl2_paired_run_plan_candidate.json",
        "integrations/continual_learning/CL3D_ANALYSIS_PLAN.md",
    ),
    "statistical_plan": ("integrations/continual_learning/CL3D_ANALYSIS_PLAN.md",),
    "instrument_scope": ("integrations/continual_learning/CL3E_MEASUREMENT_SCOPE.md",),
    "runner_and_artifacts": (
        "integrations/continual_learning/CL3B_OFFLINE_PREFLIGHT.md",
        "integrations/continual_learning/cl3b_result_contract.py",
    ),
    "human_approvals": (),
}


def review_matrix() -> dict:
    """Read draft governance and candidate evidence paths; no training/data run."""
    plan = load_candidate()
    ready = json.loads(READINESS.read_text(encoding="utf-8"))
    if (
        ready.get("decision") != "NO_GO"
        or ready.get("go_for_training") is not False
        or ready.get("compute_approved") is not False
        or ready.get("runner_enabled") is not False
        or plan.get("compute_approved") is not False
        or plan.get("runner_enabled") is not False
        or plan.get("scientific_review_approved") is not False
    ):
        raise ValueError("CL-3F only supports locked NO_GO drafts")
    blockers = ready.get("mandatory_blockers")
    if not isinstance(blockers, list) or len(blockers) != len(EVIDENCE):
        raise ValueError("unexpected scientific blocker set")
    ids = [x.get("id") for x in blockers if isinstance(x, dict)]
    if len(ids) != len(EVIDENCE) or set(ids) != set(EVIDENCE):
        raise ValueError("blocker IDs drifted from review matrix")
    matrix = []
    for blocker in blockers:
        if blocker.get("closed") is not False:
            raise ValueError("read-only gate audit cannot mark a blocker closed")
        paths = EVIDENCE[blocker["id"]]
        matrix.append({
            "blocker_id": blocker["id"],
            "status": "OPEN_REQUIRES_INDEPENDENT_REVIEW",
            "required": blocker["required"],
            "candidate_evidence_paths": list(paths),
            "candidate_evidence_files_present": all(
                (REPOSITORY / path).is_file() for path in paths
            ),
            "scientifically_validated": False,
        })
    return {
        "audit_id": "cl3f-scientific-gate-audit-v0",
        "decision": "NO_GO",
        "training_allowed": False,
        "compute_approved": False,
        "blockers_open": len(matrix),
        "matrix": matrix,
        "missing_scientific_review_approval": True,
        "missing_separate_human_compute_approval": True,
        "note": "Candidate file presence never closes a blocker or confers scientific validity.",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read-only CL-3F gate audit; no compute")
    parser.add_argument("--review-matrix", action="store_true", required=True)
    args = parser.parse_args(argv)
    if args.review_matrix:
        print(json.dumps(review_matrix(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
