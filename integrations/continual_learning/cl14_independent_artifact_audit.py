"""CL-14 independent read-only audit of already generated CL-13 artifacts.

No dataset generator, model, training, inference or network calls. Input is
an existing GitHub Actions artifact extracted by an explicitly scoped job.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

EXPECTED_SEEDS = (101, 102, 103, 104, 105)
EXPECTED_TASKS = ("T1", "T2", "T3", "T4")
EXPECTED_SOURCE_FILES = (
    "integrations/continual_learning/cl2_data_candidate.py",
    "integrations/continual_learning/cl2_shared_rule_candidate.py",
    "integrations/continual_learning/cl10g_train_only_split.py",
    "integrations/continual_learning/cl12_supplied_data_freeze_verifier.py",
)
EXPECTED_CONFIRM = "CL13_DATA_ONLY_AUTHORIZED_2026_11_02"
SPLIT_VERSION = "cl10g-train-only-stratified-hash-v1"


def _check(ok: bool, detail: str) -> None:
    if not ok:
        raise ValueError("CL-14 NO_GO: " + detail)


def _canonical(obj: object) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True).encode("ascii")


def audit_existing_cl13_artifact(folder: Path, repo: Path) -> dict:
    folder, repo = Path(folder), Path(repo)
    _check(folder.is_dir() and not folder.is_symlink(), "unsafe artifact directory")
    expected = {"manifest.json"} | {f"seed-{s}.json" for s in EXPECTED_SEEDS}
    _check({item.name for item in folder.iterdir()} == expected,
           "unexpected or missing artifact file")
    for item in folder.iterdir():
        _check(item.is_file() and not item.is_symlink() and item.stat().st_size < 300_000,
               "unsafe or oversized artifact entry")
    manifest = json.loads((folder / "manifest.json").read_text(encoding="ascii"))
    _check(type(manifest) is dict and
           manifest.get("kind") == "CL13_AUTHORIZED_DATA_ONLY_CANDIDATE" and
           manifest.get("decision") == "NO_GO" and
           manifest.get("authorization_scope") == EXPECTED_CONFIRM and
           manifest.get("generator_version") == "cl2-shared-rule-candidate-v0",
           "manifest identity drift")
    _check(manifest.get("seeds") == list(EXPECTED_SEEDS)
           and manifest.get("tasks_per_seed") == 4
           and manifest.get("train_per_task") == 128
           and manifest.get("heldout_per_task") == 64
           and manifest.get("total_records") == 3840,
           "unexpected experiment plan")
    _check(manifest.get("training_authorized") is False
           and manifest.get("inference_authorized") is False
           and manifest.get("model_modules_imported") is False
           and manifest.get("physical_energy") == "NOT_MEASURED"
           and manifest.get("scientific_signoff") is False
           and manifest.get("experiment_observations") == 0
           and manifest.get("dataset_origin_independently_authenticated") is False,
           "unapproved observation or authority claim")
    sources = manifest.get("source_sha256")
    _check(type(sources) is dict and set(sources) == set(EXPECTED_SOURCE_FILES),
           "source map mismatch")
    for name in EXPECTED_SOURCE_FILES:
        _check(hashlib.sha256((repo / name).read_bytes()).hexdigest() == sources[name],
               "source byte hash mismatch: " + name)

    file_hashes = manifest.get("file_sha256")
    reports = manifest.get("verification_reports")
    _check(type(file_hashes) is dict and set(file_hashes) == expected - {"manifest.json"},
           "data file manifest mismatch")
    _check(type(reports) is list and len(reports) == 5, "report count mismatch")
    digests = {}
    for seed, report in zip(EXPECTED_SEEDS, reports):
        name = f"seed-{seed}.json"
        raw = (folder / name).read_bytes()
        _check(hashlib.sha256(raw).hexdigest() == file_hashes[name],
               "data file checksum mismatch")
        payload = json.loads(raw)
        _check(raw == _canonical(payload) + b"\n", "non-canonical file bytes")
        _check(type(payload) is list and len(payload) == 4, "task list mismatch")
        seen = set()
        partitions = {}
        for expected_task, task in zip(EXPECTED_TASKS, payload):
            _check(type(task) is dict and set(task) == {"task_id", "train", "test"}
                   and task["task_id"] == expected_task, "task order/shape mismatch")
            for part, count in (("train", 128), ("test", 64)):
                rows = task[part]
                _check(type(rows) is list and len(rows) == count, "row count mismatch")
                label_counts = Counter()
                for row in rows:
                    _check(type(row) is list and len(row) == 2, "record shape mismatch")
                    bits, label = row
                    _check(type(bits) is list and len(bits) == 16
                           and all(type(bit) is int and bit in (0, 1) for bit in bits)
                           and type(label) is int and label in (0, 1), "nonbinary record")
                    _check(label == (bits[0] ^ bits[1]), "XOR rule mismatch")
                    identity = tuple(bits)
                    _check(identity not in seen, "duplicate feature vector")
                    seen.add(identity)
                    label_counts[label] += 1
                _check(label_counts == {0: count // 2, 1: count // 2},
                       "class balance mismatch")
            train = task["train"]
            splits = []
            for label in (0, 1):
                ranks = []
                for index, (bits, value) in enumerate(train):
                    if value == label:
                        identity = "".join(str(v) for v in bits)
                        salt = f"{SPLIT_VERSION}|{seed}|{expected_task}|{identity}".encode("ascii")
                        ranks.append((hashlib.sha256(salt).hexdigest(), identity, index))
                splits += [index for _, _, index in sorted(ranks)[:16]]
            validation = sorted(splits)
            fit = sorted(set(range(128)) - set(validation))
            partition_digest = hashlib.sha256(_canonical(
                {"fit": fit, "validation": validation}
            )).hexdigest()
            partitions[expected_task] = partition_digest
        _check(len(seen) == 768, "seed record cardinality mismatch")
        digest = hashlib.sha256(_canonical(payload)).hexdigest()
        _check(type(report) is dict and report.get("seed") == seed
               and report.get("decision") == "NO_GO"
               and report.get("dataset_sha256_candidate") == digest
               and report.get("train_only_partition_sha256_by_task") == partitions
               and report.get("train_count") == 512
               and report.get("heldout_count") == 256
               and report.get("data_origin_authenticated") is False
               and report.get("generator_executed") is False
               and report.get("training_authorized") is False,
               "reported source/split parity mismatch")
        digests[str(seed)] = digest

    return {
        "decision": "NO_GO",
        "scope": "INDEPENDENT_ARTIFACT_BYTES_AND_SEMANTICS_ONLY",
        "seed_data_sha256": digests,
        "verified_record_count": 3840,
        "source_byte_hashes_match_checkout": True,
        "heldout_tuning_performed": False,
        "model_inference_performed": False,
        "training_performed": False,
        "scientific_data_freeze_approved": False,
        "origin_authenticated_beyond_github_artifact": False,
        "physical_energy": "NOT_MEASURED",
    }


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        raise SystemExit("usage: python cl14_independent_artifact_audit.py ARTIFACT_DIR REPO_DIR")
    print(json.dumps(audit_existing_cl13_artifact(Path(sys.argv[1]), Path(sys.argv[2])),
                     sort_keys=True))
