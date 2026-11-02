"""CL-10C bounded artifact writer for future independently authorized H1 runs.

NO model imports, training, inference, dataset generation or experiment execution.
Tests use fabricated *schema fixtures* only, never experimental observations.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Mapping, Sequence

FILES = (
    "manifest.json", "attempts.jsonl", "heldout_accuracy.jsonl",
    "source_events.jsonl", "failure_ledger.jsonl",
)
MAX_BYTES = 20 * 1024 * 1024
RUN_ID = re.compile(r"^[a-z0-9][a-z0-9-]{0,95}$")
HEX256 = re.compile(r"^[0-9a-f]{64}$")
HEX40 = re.compile(r"^[0-9a-f]{40}$")
ALLOWED_STATUS = {"COMPLETE", "FAILED", "INCOMPLETE"}
ARMS = {"dense_sgd", "local_hybrid", "sparse_local_hybrid", "sparse_shared_index"}
SEEDS = {101, 102, 103, 104, 105}
TASKS = {"T1", "T2", "T3", "T4"}
CHECKPOINTS = {"pretrain", "post_T1", "post_T2", "post_T3", "post_T4"}
PHASES = {"train", "heldout_inference"}
EVENTS = (
    "logical_parameter_reads", "logical_parameter_writes", "input_visits",
    "active_index_visits", "control_events", "nonlinear_events",
)
HASH_FIELDS = (
    "protocol_sha256", "source_tree_sha256",
    "environment_manifest_sha256", "dataset_manifest_sha256",
    "units_and_metric_definitions_sha256", "instrumentation_version_sha256",
)
REQUIRED_MANIFEST = ("immutable_source_commit_sha",) + HASH_FIELDS + (
    "per_seed_dataset_sha256", "ordered_seed_list", "ordered_arm_list",
    "task_splits_and_digest_list", "review_id",
    "explicit_user_authorization_reference",
)


def _required(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(f"CL-10C NO_GO: {message}")


def _hash(value: object) -> bool:
    return type(value) is str and HEX256.fullmatch(value) is not None


def _bounded_string(value: object, *, nullable: bool = False) -> bool:
    return (nullable and value is None) or (
        type(value) is str and 0 < len(value) <= 512
        and all(ord(ch) >= 32 for ch in value)
    )


def validate_manifest(manifest: Mapping[str, object]) -> None:
    _required(isinstance(manifest, dict), "manifest must be an object")
    _required(all(key in manifest for key in REQUIRED_MANIFEST),
              "incomplete manifest provenance")
    _required(type(manifest["immutable_source_commit_sha"]) is str and
              HEX40.fullmatch(manifest["immutable_source_commit_sha"]) is not None,
              "invalid immutable commit SHA")
    for field in HASH_FIELDS:
        _required(_hash(manifest[field]), f"invalid {field}")
    _required(manifest["ordered_seed_list"] == [101, 102, 103, 104, 105],
              "seed order changed")
    _required(manifest["ordered_arm_list"] == [
        "dense_sgd", "local_hybrid", "sparse_local_hybrid", "sparse_shared_index"
    ], "arm order changed")
    digests = manifest["per_seed_dataset_sha256"]
    _required(isinstance(digests, dict) and set(digests) ==
              {str(seed) for seed in SEEDS} and all(_hash(x) for x in digests.values()),
              "per-seed digests missing")
    _required(isinstance(manifest["task_splits_and_digest_list"], list) and
              bool(manifest["task_splits_and_digest_list"]),
              "task split provenance missing")
    _required(_bounded_string(manifest["review_id"]) and
              _bounded_string(manifest["explicit_user_authorization_reference"]),
              "approval references missing")
    _required(manifest.get("physical_energy") == "NOT_MEASURED",
              "unsupported physical energy claim")
    _required(manifest.get("artifact_schema") == "cl10c-v1",
              "artifact schema changed")


def _identity(row: Mapping[str, object]) -> None:
    _required(type(row.get("seed")) is int and row["seed"] in SEEDS, "seed invalid")
    _required(row.get("arm") in ARMS, "arm invalid")
    _required(row.get("task") in TASKS, "task invalid")
    _required(row.get("phase") in PHASES, "phase invalid")
    _required(row.get("checkpoint") in CHECKPOINTS, "checkpoint invalid")


def validate_attempt(row: Mapping[str, object]) -> None:
    _required(isinstance(row, dict), "attempt must be object")
    _identity(row)
    _required(type(row.get("attempt_number")) is int and
              row["attempt_number"] >= 1, "attempt number invalid")
    _required(row.get("run_status") in ALLOWED_STATUS, "status invalid")
    reason = row.get("failure_reason")
    if row["run_status"] == "COMPLETE":
        _required(reason is None, "completed attempt has failure reason")
    else:
        _required(_bounded_string(reason), "missing failure reason")
    _required(_bounded_string(row.get("run_id")), "missing run id")
    _required(row.get("source_commit_sha") is None or
              (type(row["source_commit_sha"]) is str and
               HEX40.fullmatch(row["source_commit_sha"]) is not None),
              "invalid source_commit_sha")
    _required(row.get("observed_record_digest") is None or
              _hash(row["observed_record_digest"]),
              "invalid observed_record_digest")
    _required(row.get("artifact_digest_or_null") is None or
              _hash(row["artifact_digest_or_null"]),
              "invalid artifact digest")


def validate_accuracy(row: Mapping[str, object]) -> None:
    _required(isinstance(row, dict), "accuracy row must be object")
    _identity(row)
    _required(row["phase"] == "heldout_inference", "accuracy phase invalid")
    a, b = row.get("correct"), row.get("eligible")
    _required(type(a) is int and type(b) is int and b > 0 and 0 <= a <= b,
              "accuracy numerator/denominator invalid")


def validate_events(row: Mapping[str, object]) -> None:
    _required(isinstance(row, dict), "event row must be object")
    _identity(row)
    _required(row.get("provenance") == "OBSERVED_SOURCE_EVENTS",
              "static projection masquerades as observation")
    _required(type(row.get("denominator")) is int and row["denominator"] > 0,
              "event denominator invalid")
    _required(all(type(row.get(key)) is int and row[key] >= 0 for key in EVENTS),
              "missing/invalid observed event vector")


def _serialize(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")


def build_payloads(
    manifest: dict, attempts: Sequence[dict], accuracy: Sequence[dict],
    events: Sequence[dict], failures: Sequence[dict],
) -> dict[str, bytes]:
    """Validate complete supplied data; does not create/obtain observations."""
    validate_manifest(manifest)
    _required(all(type(rows) is list for rows in
                  (attempts, accuracy, events, failures)), "records must be lists")
    for row in attempts:
        validate_attempt(row)
    for row in accuracy:
        validate_accuracy(row)
    for row in events:
        validate_events(row)
    for row in failures:
        validate_attempt(row)
        _required(row["run_status"] != "COMPLETE",
                  "failure ledger contains successful attempt")
    _required(all(row in attempts for row in failures),
              "failure not present in attempts ledger")
    _required(all(row in failures for row in attempts
                  if row["run_status"] != "COMPLETE"),
              "failed/incomplete attempt missing in failure ledger")

    payloads = {
        "manifest.json": _serialize(manifest),
        "attempts.jsonl": b"".join(map(_serialize, attempts)),
        "heldout_accuracy.jsonl": b"".join(map(_serialize, accuracy)),
        "source_events.jsonl": b"".join(map(_serialize, events)),
        "failure_ledger.jsonl": b"".join(map(_serialize, failures)),
    }
    _required(sum(map(len, payloads.values())) <= MAX_BYTES - 4096,
              "artifact budget exceeded")
    return payloads


def write_artifact_bundle(
    output_parent: Path, run_id: str, *, manifest: dict,
    attempts: list[dict], accuracy: list[dict], events: list[dict],
    failures: list[dict],
) -> Path:
    """Publish a new bounded bundle; refuse an existing destination.

    This is a candidate local writer. Caller must enforce unique run IDs,
    access controls and durable remote immutability in a future reviewed runner.
    """
    _required(type(run_id) is str and RUN_ID.fullmatch(run_id) is not None,
              "unsafe run id")
    payloads = build_payloads(manifest, attempts, accuracy, events, failures)
    parent = Path(output_parent)
    _required(parent.is_dir() and not parent.is_symlink(),
              "existing non-symlink output parent required")
    destination = parent / run_id
    _required(not destination.exists() and not destination.is_symlink(),
              "bundle already exists")
    with tempfile.TemporaryDirectory(prefix=".cl10c-", dir=parent) as tmp:
        staging = Path(tmp) / "bundle"
        staging.mkdir()
        checksums = []
        for filename in FILES:
            data = payloads[filename]
            with (staging / filename).open("xb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            checksums.append(f"{hashlib.sha256(data).hexdigest()}  {filename}\n")
        checksums_blob = "".join(checksums).encode("ascii")
        _required(sum(map(len, payloads.values())) + len(checksums_blob) <= MAX_BYTES,
                  "checksums exceed artifact budget")
        with (staging / "checksums.sha256").open("xb") as stream:
            stream.write(checksums_blob)
            stream.flush()
            os.fsync(stream.fileno())
        _required(not destination.exists(), "bundle appeared concurrently")
        staging.rename(destination)
    return destination
