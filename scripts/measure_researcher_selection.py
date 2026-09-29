#!/usr/bin/env python3
"""Read-only measurement of paired AI Researcher experiment outcomes."""
import argparse
import json
from pathlib import Path


def summarize(record):
    if not isinstance(record, dict):
        return None
    config = record.get("configuration")
    summary = record.get("summary")
    dataset = record.get("dataset")
    if not all(isinstance(x, dict) for x in (config, summary, dataset)):
        return None
    seeds = config.get("seeds")
    epochs = config.get("epochs")
    train_shape = dataset.get("train_shape")
    test_shape = dataset.get("test_shape")
    dataset_name = dataset.get("name")
    # Without a complete protocol, even equal seeds do not establish comparable totals.
    if (not isinstance(dataset_name, str) or not dataset_name
        or not isinstance(train_shape, list) or not isinstance(test_shape, list)
        or len(train_shape) < 1 or len(test_shape) < 1
        or any(type(n) is not int or n <= 0 for n in train_shape + test_shape)
        or type(epochs) is not int or epochs <= 0
        or not isinstance(seeds, list) or not seeds
        or any(type(s) is not int or s <= 0 for s in seeds)
        or len(set(seeds)) != len(seeds)):
        return None
    # Missing fields must never compare equal merely because dict.get returns None.
    learning_rate = config.get("learning_rate")
    batch_size = config.get("batch_size")
    frequencies = config.get("frequencies")
    if (type(learning_rate) not in (int, float) or not 0 < learning_rate < float("inf")
        or type(batch_size) is not int or batch_size <= 0
        or not isinstance(frequencies, list) or not frequencies
        or any(type(v) is not int or v <= 0 for v in frequencies)):
        return None
    controls = [k for k in summary if isinstance(k, str) and k.endswith("_control")]
    if len(controls) != 1 or len(summary) < 2:
        return None
    control_name = controls[0]
    # Fail the entire file closed if ANY candidate/control is malformed.
    for metrics in summary.values():
        if (not isinstance(metrics, dict)
            or type(metrics.get("aggregate_qant_correct")) is not int
            or type(metrics.get("parameter_count")) is not int
            or metrics["aggregate_qant_correct"] < 0
            or metrics["parameter_count"] <= 0
            or metrics["aggregate_qant_correct"] > len(seeds) * test_shape[0]):
            return None
    control = summary[control_name]
    protocol = {
        "dataset_name": dataset_name,
        "train_shape": train_shape,
        "test_shape": test_shape,
        "seeds": seeds,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "frequencies": frequencies,
    }
    rows = []
    for name, metrics in summary.items():
        if name == control_name:
            continue
        correct = metrics["aggregate_qant_correct"]
        params = metrics["parameter_count"]
        margin = correct - control["aggregate_qant_correct"]
        rows.append({
            "experiment_id": record.get("experiment_id"),
            "candidate": name,
            "control": control_name,
            "seeds": seeds,
            "epochs": epochs,
            "protocol": protocol,
            "candidate_correct": correct,
            "control_correct": control["aggregate_qant_correct"],
            "correct_margin": margin,
            "candidate_parameters": params,
            "control_parameters": control["parameter_count"],
            "parameter_saving": control["parameter_count"] - params,
            "passes_relative_gate_10": margin >= -10,
        })
    return rows


def pareto(rows):
    """Compare passing observations only within an identical recorded protocol."""
    passing = [r for r in rows if r["passes_relative_gate_10"]]
    return [r for r in passing if not any(
        other is not r
        and other.get("protocol") == r.get("protocol")
        and other["candidate_correct"] >= r["candidate_correct"]
        and other["candidate_parameters"] <= r["candidate_parameters"]
        and (other["candidate_correct"] > r["candidate_correct"]
             or other["candidate_parameters"] < r["candidate_parameters"])
        for other in passing
    )]


def build_report(files):
    observations, skipped = [], []
    for path in sorted(files):
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            rows = summarize(data)
        except (OSError, ValueError, TypeError):
            rows = None
        if not rows:
            skipped.append(str(path))
            continue
        for row in rows:
            row["source_file"] = str(path)
        observations.extend(rows)
    return {
        "schema_version": 2,
        "purpose": "observational baseline; no automatic experiment selection or compute",
        "comparison_warning": "Only identical recorded protocols are Pareto-comparable; missing protocol fields are excluded. Paired concurrent control remains the primary evidence.",
        "completed_candidate_observations": len(observations),
        "passing_relative_gate_10": sum(r["passes_relative_gate_10"] for r in observations),
        "observations": observations,
        "within_protocol_pareto": pareto(observations),
        "skipped_files": skipped,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="pnn-v1/results")
    parser.add_argument("--output")
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
