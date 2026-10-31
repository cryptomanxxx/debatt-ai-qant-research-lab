"""CL-3L read-only proposed source fingerprint preview; NO experiment execution.

A preview is NOT a signed/approved provenance lock and never authorizes compute.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from integrations.continual_learning.cl2_paired_run_plan import load_candidate

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
REQUIRED_SOURCE_PATHS = (
    "integrations/continual_learning/cl2_data_candidate.py",
    "integrations/continual_learning/cl2_shared_rule_candidate.py",
    "integrations/continual_learning/cl2_mlp_baseline.py",
    "integrations/continual_learning/cl2_local_mlp.py",
    "integrations/continual_learning/cl2_paired_run_plan.py",
    "integrations/continual_learning/cl2_paired_run_plan_candidate.json",
    "integrations/continual_learning/cl3b_result_contract.py",
    "integrations/continual_learning/cl3d_analysis_contract.py",
    "integrations/continual_learning/cl3e_measurement_ledger.py",
    "integrations/continual_learning/cl3h_diagnostic_contract.py",
    "integrations/continual_learning/cl3j_artifact_contract.py",
    "integrations/continual_learning/cl3k_pair_integrity.py",
)


def fingerprint_preview() -> dict:
    """Hash a fixed allowlist of existing files; read-only, no model imports."""
    plan = load_candidate()
    hashes = {}
    for relative in REQUIRED_SOURCE_PATHS:
        path = REPO_ROOT / relative
        if not path.is_file():
            raise ValueError(f"required candidate source missing: {relative}")
        hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    canonical = json.dumps(hashes, separators=(",", ":"), sort_keys=True).encode("ascii")
    return {
        "kind": "CL3L_SOURCE_FINGERPRINT_PREVIEW_NOT_APPROVAL",
        "protocol_id": plan["protocol_id"],
        "files_sha256": hashes,
        "aggregate_allowlist_digest_sha256": hashlib.sha256(canonical).hexdigest(),
        "source_git_commit": "NOT_VERIFIED_BY_THIS_PREVIEW",
        "complete_runtime_dependency_lock": False,
        "scientific_review_approved": False,
        "compute_approved": False,
        "training_allowed": False,
        "note": "A hash preview cannot authenticate Git commit, packaging, environment, dataset content, human approval or experiment execution.",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CL-3L source fingerprint preview only")
    parser.add_argument("--preview", action="store_true", required=True)
    args = parser.parse_args(argv)
    if args.preview:
        print(json.dumps(fingerprint_preview(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
