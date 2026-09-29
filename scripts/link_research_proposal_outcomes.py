#!/usr/bin/env python3
"""Read-only, explicit proposal-to-result evidence linkage. Never authorizes compute."""
import argparse
import json
from pathlib import Path
from scripts.measure_researcher_selection import summarize


def link(manifest, result_paths):
    """Manifest entries: proposal_id, proposal_file, strategy (optional).

    Only explicit matching IDs count. A draft without a stable ID is unlinked.
    Never infer authorship from filename, candidate width or timestamps.
    """
    if not isinstance(manifest, list):
        raise ValueError("manifest must be a list")
    by_id = {}
    for entry in manifest:
        if not isinstance(entry, dict):
            raise ValueError("manifest entry must be an object")
        pid, source = entry.get("proposal_id"), entry.get("proposal_file")
        if not isinstance(pid, str) or not pid.strip() or not isinstance(source, str) or not source.strip():
            raise ValueError("proposal_id and proposal_file are required")
        if pid in by_id:
            raise ValueError("duplicate proposal_id in manifest: " + pid)
        proposal = json.loads(Path(source).read_text(encoding="utf-8"))
        if not isinstance(proposal, dict):
            raise ValueError("proposal must be an object: " + source)
        declared = proposal.get("proposal_id")
        if declared != pid:
            raise ValueError("proposal snapshot ID mismatch or missing: " + pid)
        by_id[pid] = (entry, proposal)
    matches = {pid: [] for pid in by_id}
    excluded = []
    for path in sorted(result_paths):
        try:
            result = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            excluded.append({"file": str(path), "reason": "unreadable_json"})
            continue
        if not isinstance(result, dict):
            excluded.append({"file": str(path), "reason": "invalid_root"})
            continue
        pid = result.get("proposal_id")
        if not isinstance(pid, str) or pid not in by_id:
            excluded.append({"file": str(path), "reason": "unknown_or_missing_proposal_id", "proposal_id": pid})
            continue
        entry, proposal = by_id[pid]
        rows = summarize(result)
        if not rows:
            excluded.append({"file": str(path), "reason": "incomplete_or_invalid_paired_result", "proposal_id": pid})
            continue
        design = proposal.get("proposal", proposal).get("experiment_design")
        if not isinstance(design, dict):
            excluded.append({"file": str(path), "reason": "missing_proposal_design", "proposal_id": pid})
            continue
        dataset = design.get("dataset")
        config = result["configuration"]
        if (not isinstance(dataset, dict)
            or any(dataset.get(k) != result["dataset"].get(k) for k in ("name", "train_shape", "test_shape"))
            or design.get("seeds") != config.get("seeds")
            or design.get("epochs") != config.get("epochs")):
            excluded.append({"file": str(path), "reason": "proposal_result_protocol_mismatch", "proposal_id": pid})
            continue
        proposed = design.get("candidates")
        if (not isinstance(proposed, list) or not proposed
            or any(not isinstance(c, dict) or type(c.get("local_width")) is not int for c in proposed)):
            excluded.append({"file": str(path), "reason": "unverifiable_candidate_design", "proposal_id": pid})
            continue
        widths = {c["local_width"] for c in proposed}
        recorded = config.get("candidates")
        if not isinstance(recorded, dict):
            excluded.append({"file": str(path), "reason": "missing_result_candidates", "proposal_id": pid})
            continue
        actual = {v[0] for v in recorded.values() if isinstance(v, list) and v and type(v[0]) is int}
        if len(actual) != len(recorded) or actual != (widths | {16}):
            excluded.append({"file": str(path), "reason": "candidate_width_mismatch", "proposal_id": pid})
            continue
        matches[pid].append({"result_file": str(path), "experiment_id": result.get("experiment_id"),
                             "observations": rows})
    linked = []
    for pid, (entry, _) in by_id.items():
        linked.append({"proposal_id": pid, "proposal_file": entry["proposal_file"],
                       "strategy": entry.get("strategy", "unclassified"),
                       "status": "linked" if matches[pid] else "no_verified_result",
                       "results": matches[pid]})
    return {"schema_version": 1, "purpose": "observational proposal-to-outcome lineage only; no compute or approval",
            "linked_proposals": linked, "excluded_results": excluded,
            "warning": "Strategy comparisons require preregistered equal budgets and matched protocols; this report makes no causal claim."}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--results", default="pnn-v1/results", type=Path)
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    report = link(json.loads(a.manifest.read_text(encoding="utf-8")), a.results.glob("*.json"))
    rendered = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if a.output:
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
