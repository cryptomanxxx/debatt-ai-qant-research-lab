"""Deterministic, read-only discovery index over committed research records.

This is not Research Memory, an experiment runner, or a provenance attestor.
Never import this catalog into strategy-isolated Dev-4/v3 researcher contexts.
Only stdlib; no provider/network/compute calls. Running the module's CLI writes
at most its explicitly selected derived output file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CATALOG_VERSION = 1
MAX_SOURCE_BYTES = 2_000_000
TOOLKIT_NAME = re.compile(r"exp[0-9]{3}_[a-z0-9_]+\.json\Z")
PNN_NUMBERED = re.compile(r"result([0-9]{3})\.json\Z")
HEX40 = re.compile(r"[0-9a-f]{40}\Z")
RECORD_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,159}\Z")
NON_RESEARCH_ROOT_JSON = frozenset(
    {"automation_smoke_test.json", "result_schema.example.json"}
)
EXCLUDED_FROM_DISCOVERY = (
    "research_queue/benchmarks/** (frozen strategy/holdout evidence)",
    "research_queue/human_review/** (approval/history, not experiment data)",
    "research_queue/ai_researcher/** (live proposal snapshots)",
    "local_results/** (transient/uncommitted outputs)",
    "GitHub Actions artifact contents (not committed result JSON)",
)


class CatalogError(ValueError):
    """Fail closed on an ambiguous or inconsistent source record."""


def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CatalogError("duplicate JSON key: " + key)
        result[key] = value
    return result


def _reject_constant(value: str) -> Any:
    raise CatalogError("non-standard JSON numeric constant: " + value)


def _read(root: Path, path: Path) -> tuple[dict[str, Any], dict[str, str]]:
    if path.is_symlink() or not path.is_file():
        raise CatalogError("unsafe or missing source file: " + str(path))
    try:
        relative = path.relative_to(root).as_posix()
    except ValueError as exc:
        raise CatalogError("source outside repository") from exc
    raw = path.read_bytes()
    if len(raw) > MAX_SOURCE_BYTES:
        raise CatalogError("oversized catalog source: " + relative)
    try:
        data = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_no_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CatalogError("invalid JSON in " + relative) from exc
    if not isinstance(data, dict):
        raise CatalogError("expected JSON object in " + relative)
    git_header = ("blob " + str(len(raw)) + "\0").encode("ascii")
    source = {
        "path": relative,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "git_blob_sha1": hashlib.sha1(git_header + raw).hexdigest(),
    }
    return data, source


def _require_id(value: Any, path: str) -> str:
    if not isinstance(value, str) or not RECORD_ID.fullmatch(value):
        raise CatalogError("missing or invalid identifier in " + path)
    return value


def _version(data: dict[str, Any], path: str) -> int | None:
    value = data.get("schema_version")
    if value is None:
        return None  # Do not manufacture a source schema version.
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise CatalogError("invalid source schema_version in " + path)
    return value


def _catalog_id(series: str, kind: str, record_id: str) -> str:
    return ":".join((series, kind, record_id, "v" + str(CATALOG_VERSION)))


def _dataset(value: Any, path: str) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, str) and value:
        return {"name": value}
    if isinstance(value, dict) and isinstance(value.get("name"), str):
        return {key: value[key] for key in (
            "name", "split", "train_shape", "test_shape"
        ) if key in value}
    raise CatalogError("unsupported dataset descriptor in " + path)


def _seeds(data: dict[str, Any], path: str) -> list[int] | None:
    cfg = data.get("configuration")
    if cfg is not None and not isinstance(cfg, dict):
        raise CatalogError("invalid configuration in " + path)
    candidate = cfg.get("seeds") if isinstance(cfg, dict) else None
    if candidate is None:
        candidate = data.get("seeds", data.get("seed"))
    if candidate is None:
        return None
    if isinstance(candidate, int) and not isinstance(candidate, bool):
        candidate = [candidate]
    if (not isinstance(candidate, list) or not candidate or
        any(isinstance(n, bool) or not isinstance(n, int) or n < 0
            for n in candidate) or len(set(candidate)) != len(candidate)):
        raise CatalogError("invalid, duplicate or ambiguous seeds in " + path)
    return candidate


def _proposal_files(root: Path) -> tuple[list[dict[str, Any]], dict[tuple[str, str], dict[str, str]], list[dict[str, str]]]:
    records: list[dict[str, Any]] = []
    lookup: dict[tuple[str, str], dict[str, str]] = {}
    sources: list[dict[str, str]] = []
    for series, folder in (
        ("toolkit", root / "research_queue" / "proposals"),
        ("pnn-v1", root / "pnn-v1" / "proposals"),
    ):
        for path in sorted(folder.glob("*.json")):
            data, source = _read(root, path)
            pid = _require_id(data.get("proposal_id"), source["path"])
            expected_id = path.stem if series == "toolkit" else "pnn-v1-" + path.stem
            if pid.casefold() != expected_id.casefold():
                raise CatalogError("proposal filename/id disagree: " + source["path"])
            key = (series, pid.casefold())
            if key in lookup:
                raise CatalogError("duplicate proposal identity: " + pid)
            lookup[key] = source
            sources.append(source)
            records.append({
                "catalog_id": _catalog_id(series, "proposal", pid),
                "series": series,
                "record_kind": "proposal",
                "record_id": pid,
                "status": "proposal_documented",  # Not a current job state.
                "source_schema_version": _version(data, source["path"]),
                "source": source,
                "target_experiment_id": data.get("experiment_id"),
            })
    return records, lookup, sources


def _jobs_and_executions(root: Path) -> tuple[dict[str, tuple[dict[str, Any], dict[str, str]]], dict[str, list[dict[str, Any]]], list[dict[str, str]]]:
    jobs: dict[str, tuple[dict[str, Any], dict[str, str]]] = {}
    executions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    sources: list[dict[str, str]] = []
    for path in sorted((root / "research_queue" / "jobs").glob("*.json")):
        data, source = _read(root, path)
        jid = _require_id(data.get("job_id"), source["path"])
        if jid != path.stem or jid in jobs:
            raise CatalogError("ambiguous job filename/id: " + source["path"])
        jobs[jid] = data, source
        sources.append(source)
    used_runs: set[tuple[str, str]] = set()
    for path in sorted((root / "executions").glob("*.json")):
        data, source = _read(root, path)
        jid = _require_id(data.get("job_id"), source["path"])
        stem, sep, filename_run = path.stem.rpartition("-")
        run_id = str(data.get("run_id", ""))
        if not sep or stem != jid or not run_id.isdecimal() or filename_run != run_id:
            raise CatalogError("execution filename/id mismatch: " + source["path"])
        if (jid, run_id) in used_runs:
            raise CatalogError("duplicate execution job/run identity: " + jid)
        used_runs.add((jid, run_id))
        commit = data.get("git_commit")
        if not isinstance(commit, str) or not HEX40.fullmatch(commit):
            raise CatalogError("execution lacks recorded 40-hex Git commit: " + source["path"])
        if jid not in jobs:
            raise CatalogError("execution lacks committed job definition: " + source["path"])
        executions[jid].append({
            "source": source,
            "run_id": run_id,
            "recorded_status": data.get("status"),
            "recorded_execution_commit": commit,
            "recorded_runner": data.get("runner"),
        })
        sources.append(source)
    for runs in executions.values():
        runs.sort(key=lambda row: (int(row["run_id"]), row["source"]["path"]))
    return jobs, executions, sources


def _resolve_job(
    series: str,
    path: Path,
    result: dict[str, Any],
    jobs: dict[str, tuple[dict[str, Any], dict[str, str]]],
) -> tuple[str | None, str]:
    declared = result.get("job_id")
    if declared is not None:
        jid = _require_id(declared, path.as_posix())
        if jid not in jobs:
            raise CatalogError("declared job has no committed definition: " + jid)
        return jid, "explicit_result_job_id"
    if series != "pnn-v1":
        return None, "no_declared_job"
    match = re.fullmatch(r"PNN-v1-Exp([0-9]{3})", result["experiment_id"])
    if match:
        jid = "pnn-v1-exp" + match.group(1)
        if jid not in jobs:
            raise CatalogError("numbered PNN result lacks committed job: " + jid)
        return jid, "numbered_experiment_job_convention_only"
    proposal = result.get("proposal_id")
    if isinstance(proposal, str) and proposal in jobs:
        return proposal, "auxiliary_proposal_job_name_convention_only"
    if path.stem.startswith("result_w"):
        candidate = "ai-" + path.stem.removeprefix("result_").replace("_", "-")
        if candidate in jobs:
            return candidate, "legacy_result_filename_convention_only"
    return None, "unlinked_auxiliary_result"


def _result_record(
    root: Path,
    path: Path,
    series: str,
    proposals: dict[tuple[str, str], dict[str, str]],
    jobs: dict[str, tuple[dict[str, Any], dict[str, str]]],
    executions: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    data, source = _read(root, path)
    experiment_id = _require_id(data.get("experiment_id"), source["path"])
    if series == "toolkit":
        if experiment_id != path.stem or not TOOLKIT_NAME.fullmatch(path.name):
            raise CatalogError("Toolkit filename/experiment_id mismatch: " + source["path"])
    else:
        if not experiment_id.startswith("PNN-v1-"):
            raise CatalogError("non-PNN experiment inside PNN source: " + source["path"])
        numbered = PNN_NUMBERED.fullmatch(path.name)
        if numbered and experiment_id != "PNN-v1-Exp" + numbered.group(1):
            raise CatalogError("PNN numbered filename/experiment_id mismatch: " + source["path"])
        if not numbered and not path.name.startswith("result_"):
            raise CatalogError("unknown PNN result naming convention: " + source["path"])
    backend = data.get("backend")
    if not isinstance(backend, str) or not backend:
        raise CatalogError("result has no recorded backend: " + source["path"])
    dataset = _dataset(data.get("dataset"), source["path"])
    if dataset is None:
        raise CatalogError("result has no dataset: " + source["path"])
    proposal_id = data.get("source_proposal", data.get("proposal_id"))
    if proposal_id is not None:
        proposal_id = _require_id(proposal_id, source["path"])
    proposal_source = proposals.get((series, proposal_id.casefold())) if proposal_id else None
    jid, link_basis = _resolve_job(series, path, data, jobs)
    job_source = jobs[jid][1] if jid is not None else None
    if jid:
        job_proposal = jobs[jid][0].get("source_proposal")
        if job_proposal and proposal_id and job_proposal.casefold() != proposal_id.casefold():
            raise CatalogError("result/job proposal mismatch: " + source["path"])
    runs = executions.get(jid, []) if jid else []
    if any(item["recorded_status"] != "completed" for item in runs):
        raise CatalogError("completed result linked to noncompleted execution: " + source["path"])
    cfg = data.get("configuration") or {}
    if not isinstance(cfg, dict):
        raise CatalogError("invalid result configuration: " + source["path"])
    return {
        "catalog_id": _catalog_id(series, "result", experiment_id),
        "series": series,
        "record_kind": "result",
        "record_id": experiment_id,
        "status": "completed_recorded",  # Presence of committed result, not independent audit.
        "source_schema_version": _version(data, source["path"]),
        "source": source,
        "mirrored_sources": [],
        "timestamp_utc": data.get("timestamp_utc"),
        "backend": backend,
        "dataset": dataset,
        "seeds": _seeds(data, source["path"]),
        "method": {key: cfg[key] for key in (
            "epochs", "learning_rate", "batch_size", "frequencies",
            "train_samples", "test_samples",
        ) if key in cfg},
        "proposal_id": proposal_id,
        "proposal_source": proposal_source,
        "job_id": jid,
        "job_source": job_source,
        "job_link_basis": link_basis,
        "execution_records": runs,
        "result_artifact_independently_authenticated": False,
        "source_note": "A committed result JSON is not by itself an externally authenticated raw run artifact; execution linkage is a recorded ID/convention, not a cryptographic binding to result bytes.",
    }


def _v3_planning(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    records: list[dict[str, Any]] = []
    sources: list[dict[str, str]] = []
    registry_path = root / "research_queue" / "v3_hypotheses" / "registry.json"
    if registry_path.is_file():
        data, source = _read(root, registry_path)
        if (data.get("record_kind") != "unexecuted_hypothesis_registry" or
            data.get("training_runs") != 0 or not isinstance(data.get("entries"), list)):
            raise CatalogError("v3 hypothesis registry is not an unexecuted proposal-only record")
        sources.append(source)
        for entry in data["entries"]:
            if not isinstance(entry, dict):
                raise CatalogError("malformed v3 hypothesis entry")
            hid = _require_id(entry.get("hypothesis_id"), source["path"])
            if entry.get("state") != "proposed_untested":
                raise CatalogError("v3 hypothesis status changed: " + hid)
            origin = entry.get("source")
            if not isinstance(origin, dict):
                raise CatalogError("hypothesis missing source reference: " + hid)
            records.append({
                "catalog_id": _catalog_id("v3", "hypothesis", hid),
                "series": "v3",
                "record_kind": "hypothesis",
                "record_id": hid,
                "status": "proposed_unexecuted",
                "source_schema_version": _version(data, source["path"]),
                "source": source,
                "origin_workflow_run_id": origin.get("workflow_run_id"),
                "origin_report_sha256_claim": origin.get("original_report_sha256"),
                "result": None,
            })
    draft_path = root / "research_queue" / "v3_protocols" / "first_standalone_draft.json"
    if draft_path.is_file():
        data, source = _read(root, draft_path)
        execution = data.get("execution")
        if (data.get("status") != "draft_unapproved_unexecuted" or
            data.get("record_kind") != "standalone_v3_protocol_draft" or
            not isinstance(execution, dict) or execution.get("status") != "not_authorized" or
            execution.get("training_runs") != 0 or execution.get("approval_record") is not None):
            raise CatalogError("v3 standalone record is not an unapproved/unexecuted draft")
        pid = _require_id(data.get("protocol_id"), source["path"])
        sources.append(source)
        records.append({
            "catalog_id": _catalog_id("v3", "protocol_draft", pid),
            "series": "v3",
            "record_kind": "protocol_draft",
            "record_id": pid,
            "status": "draft_unapproved_unexecuted",
            "source_schema_version": _version(data, source["path"]),
            "source": source,
            "dataset": _dataset(data.get("dataset"), source["path"]),
            "result": None,
        })
    return records, sources


def build_catalog(root: Path = ROOT) -> dict[str, Any]:
    """Derive a human-facing index. Never mutate source data or agent memory."""
    root = root.resolve()
    records, proposals, consumed = _proposal_files(root)
    jobs, executions, support_sources = _jobs_and_executions(root)
    consumed += support_sources
    canonical: dict[str, dict[str, Any]] = {}
    numbered_toolkit = sorted((root / "results").glob("exp*.json"))
    for path in numbered_toolkit:
        if not TOOLKIT_NAME.fullmatch(path.name):
            raise CatalogError("unclassified Toolkit result filename: " + path.name)
        record = _result_record(root, path, "toolkit", proposals, jobs, executions)
        if record["catalog_id"] in canonical:
            raise CatalogError("duplicate canonical result ID: " + record["catalog_id"])
        canonical[record["catalog_id"]] = record
        consumed.append(record["source"])
    pnn_by_filename: dict[str, dict[str, Any]] = {}
    for path in sorted((root / "pnn-v1" / "results").glob("*.json")):
        record = _result_record(root, path, "pnn-v1", proposals, jobs, executions)
        if record["catalog_id"] in canonical or path.name in pnn_by_filename:
            raise CatalogError("duplicate canonical PNN identity: " + record["catalog_id"])
        canonical[record["catalog_id"]] = record
        pnn_by_filename[path.name] = record
        consumed.append(record["source"])
    aliases = 0
    for path in sorted((root / "results").glob("*.json")):
        if TOOLKIT_NAME.fullmatch(path.name) or path.name in NON_RESEARCH_ROOT_JSON:
            continue
        if not path.name.startswith("result"):
            raise CatalogError("unclassified root results JSON: " + path.name)
        if path.name not in pnn_by_filename:
            raise CatalogError("orphan root PNN mirror; no canonical PNN result: " + path.name)
        mirror_data, mirror_source = _read(root, path)
        entry = pnn_by_filename[path.name]
        if mirror_data.get("experiment_id") != entry["record_id"] or mirror_source["sha256"] != entry["source"]["sha256"]:
            raise CatalogError("divergent PNN root mirror: " + path.name)
        entry["mirrored_sources"].append(mirror_source)
        consumed.append(mirror_source)
        aliases += 1
    records.extend(canonical.values())
    planning, planning_sources = _v3_planning(root)
    records.extend(planning)
    consumed.extend(planning_sources)
    identity = [row["catalog_id"] for row in records]
    if len(identity) != len(set(identity)):
        raise CatalogError("duplicate global composite catalog identity")
    records.sort(key=lambda row: row["catalog_id"])
    source_map: dict[str, str] = {}
    for source in consumed:
        p = source["path"]
        if p in source_map and source_map[p] != source["sha256"]:
            raise CatalogError("source changed during index build: " + p)
        source_map[p] = source["sha256"]
    manifest = json.dumps(sorted(source_map.items()), separators=(",", ":"), ensure_ascii=False).encode()
    associated_jobs = {item["job_id"] for item in canonical.values() if item["job_id"]}
    counts = Counter(row["record_kind"] for row in records)
    return {
        "schema_version": CATALOG_VERSION,
        "catalog_kind": "debatt_ai_global_experiment_discovery",
        "purpose": "Read-only repository inventory for human/audit discovery; NOT a researcher tool, verified raw-artifact attestation, holdout set, or compute authorization.",
        "source_manifest_sha256": hashlib.sha256(manifest).hexdigest(),
        "scope": {
            "included": [
                "results/expNNN_*.json (Toolkit canonical completed records)",
                "pnn-v1/results/*.json (PNN-v1 canonical completed records)",
                "results/result*.json (exact PNN mirrors, never new experiments)",
                "pnn-v1/proposals/*.json and research_queue/proposals/*.json (proposal documents)",
                "research_queue/v3_hypotheses/registry.json (hypothesis metadata only)",
                "research_queue/v3_protocols/first_standalone_draft.json (unexecuted draft metadata)",
                "research_queue/jobs/*.json and executions/*.json (recorded linkage metadata)",
            ],
            "excluded": list(EXCLUDED_FROM_DISCOVERY),
            "model_access_policy": "Do not feed this global index or its pointers to Dev-4 strategy selectors or strategy-isolated v3 Research Memory. Their frozen, independent allowlists/provenance and budgets are unchanged.",
            "photonic_performance_claims": False,
            "timestamp_policy": "No wall-clock build timestamp; source_manifest_sha256 and per-source digests identify the inspected bytes.",
        },
        "summary": {
            "records_by_kind": dict(sorted(counts.items())),
            "canonical_completed_result_records": len(canonical),
            "exact_pnn_mirror_files": aliases,
            "linked_job_ids": len(associated_jobs),
            "execution_records_without_catalog_result": sorted(
                source["path"] for jid, runs in executions.items() if jid not in associated_jobs
                for source in (item["source"] for item in runs)
            ),
        },
        "records": records,
    }


def canonical_bytes(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False,
                       allow_nan=False) + "\n").encode("utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT, help="repository checkout")
    parser.add_argument("--output", type=Path, default=Path("public/experiment-catalog.json"),
                        help="derived JSON output (relative to --root unless absolute)")
    parser.add_argument("--check", action="store_true", help="verify existing snapshot without writing")
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    output = output.resolve()
    for source_dir in ("results", "pnn-v1", "research_queue", "experiments", "executions"):
        if output.is_relative_to(root / source_dir):
            parser.error("output must not overwrite research source/evidence directories")
    content = canonical_bytes(build_catalog(root))
    if args.check:
        if not output.is_file() or output.read_bytes() != content:
            raise SystemExit("Experiment catalog is absent/stale: regenerate the derived output")
        print("Catalog matches committed source bytes:", output)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(content)
    print("Wrote read-only experiment discovery catalog:", output)


if __name__ == "__main__":
    main()
