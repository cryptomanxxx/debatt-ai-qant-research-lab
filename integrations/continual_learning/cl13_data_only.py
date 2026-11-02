"""CL-13: explicitly Marcus-authorized H1 DATA ONLY generation.

Generates exactly seeds 101..105 and four tasks, validates all records,
writes content hashes. No model import, train, inference or energy measurement.
This CLI is gated to an exact opt-in string and explicit output path.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

# Import dataset generator only; no model modules.
from integrations.continual_learning.cl2_shared_rule_candidate import (
    VERSION as GENERATOR_VERSION,
    generate_shared_rule_candidate,
    shared_rule_dataset_sha256,
)
from integrations.continual_learning.cl12_supplied_data_freeze_verifier import (
    inspect_supplied_seed_records,
)

SEEDS = (101, 102, 103, 104, 105)
CONFIRMATION = "CL13_DATA_ONLY_AUTHORIZED_2026_11_02"
SOURCE_FILES = (
    "integrations/continual_learning/cl2_data_candidate.py",
    "integrations/continual_learning/cl2_shared_rule_candidate.py",
    "integrations/continual_learning/cl10g_train_only_split.py",
    "integrations/continual_learning/cl12_supplied_data_freeze_verifier.py",
)
MODEL_MODULE_STEMS = (
    "cl2_mlp_baseline", "cl2_local_mlp", "cl2_sparse_forward",
    "cl2_shared_index_forward", "cl10b_gated_h1_runner",
)


def _check_no_models() -> None:
    if any(any(mod == stem or mod.endswith("." + stem)
               for stem in MODEL_MODULE_STEMS) for mod in sys.modules):
        raise RuntimeError("CL-13: forbidden model module imported")


def generate_verified_only(output_dir: Path, confirmation: str) -> dict:
    if confirmation != CONFIRMATION:
        raise PermissionError("CL-13: exact dataset-only authorization required")
    if os.environ.get("CL13_DATA_ONLY") != "1":
        raise PermissionError("CL-13: explicit environment guard required")
    _check_no_models()
    root = Path(__file__).resolve().parents[2]
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError("CL-13: output directory must not exist")
    output_dir.mkdir(parents=True)
    hashes = {}
    reports = []
    for seed in SEEDS:
        splits = generate_shared_rule_candidate(seed, train_per_task=128, test_per_task=64)
        payload = [{
            "task_id": item.task_id,
            "train": [[list(e.features), e.label] for e in item.train],
            "test": [[list(e.features), e.label] for e in item.test],
        } for item in splits]
        report = inspect_supplied_seed_records(seed, payload)
        digest = shared_rule_dataset_sha256(splits)
        if digest != report["dataset_sha256_candidate"]:
            raise RuntimeError("CL-13: independent canonical digest disagreement")
        data_file = f"seed-{seed}.json"
        raw = (json.dumps(payload, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=True) + "\n").encode("ascii")
        (output_dir / data_file).write_bytes(raw)
        hashes[data_file] = hashlib.sha256(raw).hexdigest()
        reports.append(report)
        _check_no_models()

    provenance = {}
    for filename in SOURCE_FILES:
        provenance[filename] = hashlib.sha256((root / filename).read_bytes()).hexdigest()
    manifest = {
        "kind": "CL13_AUTHORIZED_DATA_ONLY_CANDIDATE",
        "decision": "NO_GO",
        "generator_version": GENERATOR_VERSION,
        "authorization_scope": CONFIRMATION,
        "seeds": list(SEEDS),
        "tasks_per_seed": 4,
        "train_per_task": 128,
        "heldout_per_task": 64,
        "total_records": 3840,
        "file_sha256": hashes,
        "source_sha256": provenance,
        "verification_reports": reports,
        "dataset_origin_independently_authenticated": False,
        "scientific_signoff": False,
        "model_modules_imported": False,
        "training_authorized": False,
        "inference_authorized": False,
        "physical_energy": "NOT_MEASURED",
        "experiment_observations": 0,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    _check_no_models()
    return manifest


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: python -m integrations.continual_learning.cl13_data_only OUTPUT DIR_CONFIRMATION")
    manifest = generate_verified_only(Path(sys.argv[1]), sys.argv[2])
    print(json.dumps({
        "decision": manifest["decision"],
        "seeds": manifest["seeds"],
        "total_records": manifest["total_records"],
        "training_authorized": manifest["training_authorized"],
        "physical_energy": manifest["physical_energy"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
