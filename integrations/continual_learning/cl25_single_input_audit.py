"""CL-25 one archived sample provenance check, without model imports or inference."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PACKET = HERE / "cl25_single_archived_input.json"
SOURCE = "integrations/continual_learning/frozen_data_candidate/cl13/seed-101.json"
SHA256 = "308a407d08f4591784b8ba2265bd72ddde29c85d40d3dbded251bc3653f8b93e"
FEATURES = [0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
EXPECTED = {
    "kind": "CL25_SINGLE_ARCHIVED_INPUT_REVIEW_ONLY",
    "decision": "NO_GO", "review_only": True,
    "source_dataset": SOURCE, "source_dataset_canonical_sha256": SHA256,
    "seed": 101, "task": "T1", "partition": "train", "index": 0,
    "features": FEATURES, "label": 0, "heldout_data_packaged": False,
    "model_import_authorized": False, "inference_authorized": False,
    "training_authorized": False, "energy_measurement_authorized": False,
}


def validate_packaged(packet: dict) -> dict:
    if type(packet) is not dict or packet != EXPECTED:
        raise ValueError("CL-25 NO_GO: packaged input/authorization drift")
    return {
        "decision": "NO_GO", "packaged_samples": 1, "heldout_data_packaged": False,
        "model_inference_authorized": False, "models_imported": 0,
        "training_steps": 0, "physical_energy": "NOT_MEASURED",
    }


def verify_source_bytes(data: bytes, packet: dict) -> dict:
    validate_packaged(packet)
    archive = json.loads(data)
    canonical = json.dumps(
        archive, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    if hashlib.sha256(canonical).hexdigest() != SHA256:
        raise ValueError("CL-25 NO_GO: archived canonical dataset fingerprint drift")
    if data != canonical + bytes([10]):
        raise ValueError("CL-25 NO_GO: non-canonical archived file bytes")
    if (type(archive) is not list or len(archive) < 1
            or archive[0].get("task_id") != "T1"
            or archive[0]["train"][0] != [FEATURES, 0]):
        raise ValueError("CL-25 NO_GO: frozen T1 sample drift")
    return validate_packaged(packet)


def audit_current_files() -> dict:
    packet = json.loads(PACKET.read_text(encoding="ascii"))
    return verify_source_bytes((ROOT / SOURCE).read_bytes(), packet)


if __name__ == "__main__":
    print(json.dumps(audit_current_files(), sort_keys=True))
