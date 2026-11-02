"""CL-16 immutable-receipt comparison of archived CL-13 bytes. NO MODEL.

This does not approve scientific freeze, execution or source provenance beyond
the independently audited H1 artifact and its GitHub Actions identity.
"""
from __future__ import annotations

import json
from pathlib import Path

from cl14_independent_artifact_audit import audit_existing_cl13_artifact

HERE = Path(__file__).resolve().parent


def audit_repository_archive() -> dict:
    receipt = json.loads((HERE / "cl15_h1_data_lock_candidate.json").read_text(encoding="ascii"))
    audited = audit_existing_cl13_artifact(
        HERE / "frozen_data_candidate" / "cl13",
        HERE.parents[1],
    )
    if (audited["seed_data_sha256"] != receipt["seed_dataset_sha256"]
            or audited["verified_record_count"] != receipt["records_total"]
            or audited["decision"] != "NO_GO"
            or receipt["scientific_data_freeze_approved"] is not False
            or receipt["training_authorized"] is not False
            or receipt["inference_authorized"] is not False):
        raise ValueError("CL-16 NO_GO: archived dataset diverges from pin or scope")
    return {
        "decision": "NO_GO",
        "archived_seed_files": 5,
        "verified_record_count": audited["verified_record_count"],
        "seed_data_sha256": audited["seed_data_sha256"],
        "original_bytes_checked_against_reviewed_CL15_receipt": True,
        "scientific_data_freeze_approved": False,
        "training_authorized": False,
        "inference_authorized": False,
        "physical_energy": "NOT_MEASURED",
    }


if __name__ == "__main__":
    print(json.dumps(audit_repository_archive(), sort_keys=True))
