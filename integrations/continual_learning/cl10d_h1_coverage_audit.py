"""CL-10D: read-only coverage and integrity audit of a supplied H1 bundle.

NO_GO: no training, inference, dataset creation, source-event measurement or
execution authorization. A passing audit verifies structure, not authenticity.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from cl10c_artifact_writer import (
    ARMS, CHECKPOINTS, FILES, MAX_BYTES, SEEDS, TASKS,
    validate_accuracy, validate_attempt, validate_events, validate_manifest,
)

EXPECTED_TRAIN = {(arm, seed, task) for arm in ARMS for seed in SEEDS for task in TASKS}
EXPECTED_HELDOUT = {
    (arm, seed, checkpoint, task)
    for arm in ARMS for seed in SEEDS
    for checkpoint in CHECKPOINTS for task in TASKS
}


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError("CL-10D NO_GO: " + message)


def _unique_keys(records: list[dict], fields: tuple[str, ...], name: str) -> set:
    keys = [tuple(row[field] for field in fields) for row in records]
    _check(len(keys) == len(set(keys)), "duplicate " + name)
    return set(keys)


def validate_complete_h1(
    manifest: dict, attempts: list[dict], accuracy: list[dict],
    events: list[dict], failures: list[dict],
) -> dict:
    """Audit a claimed COMPLETE 20-pair H1 run, never execute one.

    A failed/incomplete experiment must remain labeled incomplete; it cannot
    satisfy this complete-coverage gate. Source events may be absent, but
    absence must be reported and prevents any event-based scientific claim.
    """
    validate_manifest(manifest)
    _check(all(type(items) is list for items in
               (attempts, accuracy, events, failures)), "ledgers must be lists")
    _check(len(attempts) == 80, "expected exactly 80 training attempt records")
    _check(len(accuracy) == 400, "expected exactly 400 heldout accuracy records")
    _check(not failures, "failed or incomplete run cannot pass complete coverage")
    run_ids = set()
    for row in attempts:
        validate_attempt(row)
        _check(row["phase"] == "train" and row["checkpoint"] == "post_" + row["task"],
               "invalid training attempt checkpoint")
        _check(row["run_status"] == "COMPLETE" and row["attempt_number"] == 1,
               "incomplete/retried training attempt")
        _check(row["source_commit_sha"] == manifest["immutable_source_commit_sha"],
               "attempt source commit mismatch")
        run_ids.add(row["run_id"])
    _check(len(run_ids) == 1, "attempt run identifiers differ")
    train_keys = _unique_keys(attempts, ("arm", "seed", "task"), "training attempt")
    _check(train_keys == EXPECTED_TRAIN, "training matrix incomplete or unexpected")
    for row in accuracy:
        validate_accuracy(row)
        _check(row["eligible"] == 64, "unexpected heldout size")
    accuracy_keys = _unique_keys(
        accuracy, ("arm", "seed", "checkpoint", "task"), "accuracy record"
    )
    _check(accuracy_keys == EXPECTED_HELDOUT, "heldout matrix incomplete or unexpected")
    for row in events:
        validate_events(row)
        _check(row["checkpoint"] == "post_" + row["task"]
               if row["phase"] == "train" else
               row["checkpoint"] in CHECKPOINTS,
               "invalid event checkpoint")
    event_keys = _unique_keys(
        events, ("arm", "seed", "phase", "checkpoint", "task"), "source event"
    )
    if events:
        expected_events = {
            (arm, seed, "train", "post_" + task, task)
            for arm, seed, task in EXPECTED_TRAIN
        } | {
            (arm, seed, "heldout_inference", checkpoint, task)
            for arm, seed, checkpoint, task in EXPECTED_HELDOUT
        }
        _check(event_keys == expected_events,
               "partial or unexpected observed source-event matrix")
    return {
        "decision": "NO_GO",
        "coverage": "COMPLETE_STRUCTURE_ONLY",
        "training_attempts": len(attempts),
        "heldout_records": len(accuracy),
        "source_event_records": len(events),
        "source_event_claim_allowed": bool(events),
        "authenticity_verified": False,
        "training_authorized": False,
    }


def audit_existing_bundle(bundle: Path) -> dict:
    """Read only an existing local bundle, verify bytes/checksums before parsing."""
    root = Path(bundle)
    _check(root.is_dir() and not root.is_symlink(), "invalid bundle directory")
    names = {item.name for item in root.iterdir()}
    _check(names == set(FILES) | {"checksums.sha256"}, "unexpected bundle files")
    for name in names:
        entry = root / name
        _check(entry.is_file() and not entry.is_symlink(), "unsafe bundle entry")
    _check(sum((root / name).stat().st_size for name in names) <= MAX_BYTES,
           "bundle too large")
    checksum_lines = (root / "checksums.sha256").read_text(encoding="ascii").splitlines()
    _check(len(checksum_lines) == len(FILES), "checksum count mismatch")
    _check(all(line == hashlib.sha256((root / name).read_bytes()).hexdigest()
               + "  " + name for line, name in zip(checksum_lines, FILES)),
           "checksum mismatch or unexpected order")

    def read_jsonl(name: str) -> list[dict]:
        return [json.loads(line) for line in
                (root / name).read_text(encoding="ascii").splitlines()]

    manifest = json.loads((root / "manifest.json").read_text(encoding="ascii"))
    return validate_complete_h1(
        manifest, read_jsonl("attempts.jsonl"),
        read_jsonl("heldout_accuracy.jsonl"), read_jsonl("source_events.jsonl"),
        read_jsonl("failure_ledger.jsonl"),
    )
