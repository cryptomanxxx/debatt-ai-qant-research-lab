#!/usr/bin/env python3
"""Offline, read-only measurement baseline for AI Researcher experiment selection.

Only structured completed results count as evidence. This script does not call an
LLM, select jobs for execution, approve proposals, or modify research history.
"""
import argparse
import json
from pathlib import Path


def summarize(record):
    config = record.get("configuration") or {}
    summary = record.get("summary") or {}
    if not isinstance(summary, dict) or not isinstance(config, dict):
        return None
    control_name = next((k for k in summary if k.endswith("_control") and isinstance(summary[k], dict)), None)
    if not control_name:
        return None
    control = summary[control_name]
    control_correct = control.get("aggregate_qant_correct")
    control_params = control.get("parameter_count")
    seeds = config.get("seeds")
    if not isinstance(control_correct, int) or isinstance(control_correct, bool) or not isinstance(control_params, int):
        return None
    if not isinstance(seeds, list) or not seeds or any(type(s) is not int for s in seeds):
        return None
    rows = []
    for name, metrics in summary.items():
        if name == control_name or not isinstance(metrics, dict):
            continue
        correct = metrics.get("aggregate_qant_correct")
        params = metrics.get("parameter_count")
        if type(correct) is not int or type(params) is not int or params <= 0:
            continue
        margin = correct - control_correct
        rows.append({
            "experiment_id": record.get("experiment_id"),
            "candidate": name,
            "control": control_name,
            "seeds": seeds,
            "epochs": config.get("epochs"),
            "candidate_correct": correct,
            "control_correct": control_correct,
            "correct_margin": margin,
            "candidate_parameters": params,
            "control_parameters": control_params,
            "parameter_saving": control_params - params,
            "passes_relative_gate_10": margin >= -10,
        })
    return rows or None


def pareto(rows):
    """Non-dominated passing observations: maximize correct and minimize params.

    This is an exploratory cross-run view, NOT a controlled statistical comparison
    across different seeds. The paired relative gate is the primary observation.
    """
    passing = [r for r in rows if r["passes_relative_gate_10"]]
    return [r for r in passing if not any(
        other is not r
        and other["candidate_correct"] >= r["candidate_correct"]
        and other["candidate_parameters"] <= r["candidate_parameters"]
        and (other["candidate_correct"] > r["candidate_correct"]
             or other["candidate_parameters"] < r["candidate_parameters"])
        and other["seeds"] == r["seeds"]
        for other in passing
    )]


def build_report(files):
    observations = []
    skipped = []
    for path in sorted(files):
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            rows = summarize(data)
        except (OSError, ValueError, TypeError):
            rows = None
        if rows is None:
            skipped.append(str(path))
        else:
            for row in rows:
                row["source_file"] = str(path)
            observations.extend(rows)
    return {
        "schema_version": 1,
        "purpose": "observational baseline; no automatic experiment selection or compute",
        "comparison_warning": "Cross-seed aggregate correct counts are not directly comparable. Use paired concurrent controls; exploratory Pareto comparison is restricted to identical seed lists.",
        "completed_candidate_observations": len(observations),
        "passing_relative_gate_10": sum(r["passes_relative_gate_10"] for r in observations),
        "observations": observations,
        "within_seed_pareto": pareto(observations),
        "skipped_files": skipped,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="pnn-v1/results", help="Directory containing completed JSON results")
    parser.add_argument("--output", help="Optional output path; otherwise print JSON to stdout")
    args = parser.parse_args()
    report = build_report(Path(args.results).glob("*.json"))
    rendered = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
