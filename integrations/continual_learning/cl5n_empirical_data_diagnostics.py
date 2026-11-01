"""CL-5N: read-only diagnostics for a previously generated CL-5M report.

This module NEVER imports or calls a data generator, model, or energy estimator.
It consumes an existing aggregate JSON report and renders derived statistics.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

EXPECTED = {(f"T{task}", split): count
            for task in range(1, 5)
            for split, count in (("train", 128), ("test", 64))}


def _integer(value: object, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"invalid nonnegative integer: {name}")
    return value


def analyze(report: dict) -> list[dict]:
    """Fail closed on wrong provenance, scope, partitions, or counters."""
    if not isinstance(report, dict):
        raise ValueError("report must be a JSON object")
    required = {
        "kind": "CL5M_ACTUAL_SINGLE_SEED_DATA_ONLY_OBSERVATION",
        "authorized_scope": "synthetic data validation only",
        "decision": "NO_GO",
        "seed": 101,
        "record_digests_match": True,
        "dataset_generation_executed": True,
        "model_training_executed": False,
        "model_inference_executed": False,
        "hardware_energy_measured": False,
        "scientific_gates_closed": 0,
        "scientific_approval_granted": False,
    }
    if any(type(report.get(k)) is not type(v) or report.get(k) != v
           for k, v in required.items()):
        raise ValueError("unexpected CL-5M provenance or safety flags")
    digest = report.get("original_dataset_sha256")
    if (not isinstance(digest, str) or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest)
            or report.get("instrumented_dataset_sha256") != digest):
        raise ValueError("dataset digest mismatch or malformed digest")
    partitions = report.get("partitions")
    counters = report.get("instrumented_counters")
    if not isinstance(partitions, list) or not isinstance(counters, list):
        raise ValueError("missing partition/counter records")
    if len(partitions) != 8 or len(counters) != 8:
        raise ValueError("expected eight partitions and eight counters")
    rows = {}
    events = {}
    for item in partitions:
        if not isinstance(item, dict):
            raise ValueError("invalid partition")
        key = (item.get("task_id"), item.get("partition"))
        if key not in EXPECTED or key in rows:
            raise ValueError("unexpected or repeated partition")
        count = _integer(item.get("count"), "count")
        if count != EXPECTED[key]:
            raise ValueError("unexpected partition size")
        zero = _integer(item.get("class_zero"), "class_zero")
        one = _integer(item.get("class_one"), "class_one")
        if zero + one != count or zero != one:
            raise ValueError("class-balance mismatch")
        rate = item.get("observed_activity_rate")
        nominal = item.get("nominal_bernoulli_rate")
        bits = item.get("mean_active_bits")
        if any(type(v) not in (float, int) or not math.isfinite(v)
               for v in (rate, nominal, bits)):
            raise ValueError("invalid activity measurement")
        if not (0 <= rate <= 1 and 0 <= nominal <= 1 and 0 <= bits <= 16):
            raise ValueError("activity out of bounds")
        if not math.isclose(rate, bits / 16, abs_tol=1e-12):
            raise ValueError("activity and active bit count disagree")
        if not math.isclose(nominal, int(key[0][1:]) / 8, abs_tol=1e-12):
            raise ValueError("unexpected task probability")
        rows[key] = item
    for item in counters:
        if not isinstance(item, dict):
            raise ValueError("invalid counters")
        key = (item.get("task_id"), item.get("partition"))
        if key not in EXPECTED or key in events:
            raise ValueError("unexpected or repeated counter partition")
        if (item.get("kind") != "CL5L_INSTRUMENTED_GENERATOR_PARTITION"
                or item.get("generator_connected") is not True
                or item.get("dataset_generation_executed") is not True
                or item.get("model_training_executed") is not False
                or item.get("attempt_cap_exhausted") is not False
                or item.get("scientific_decision") != "NO_GO"):
            raise ValueError("counter provenance mismatch")
        accepted = _integer(item.get("accepted_records"), "accepted_records")
        attempts = _integer(item.get("candidate_attempts"), "candidate_attempts")
        class_reject = _integer(item.get("class_full_rejections"), "class rejects")
        duplicate = _integer(item.get("duplicate_rejections_after_class_check"), "duplicates")
        if accepted != EXPECTED[key] or attempts != accepted + class_reject + duplicate:
            raise ValueError("counter conservation failure")
        events[key] = item
    if rows.keys() != events.keys() or rows.keys() != EXPECTED.keys():
        raise ValueError("partition/counter coverage mismatch")
    if _integer(report.get("records_checked"), "records_checked") != sum(
            row["count"] for row in rows.values()):
        raise ValueError("total record count mismatch")
    summary = []
    for task in range(1, 5):
        name = f"T{task}"
        selected = [rows[(name, split)] for split in ("train", "test")]
        selected_events = [events[(name, split)] for split in ("train", "test")]
        count = sum(row["count"] for row in selected)
        summary.append({
            "task": name,
            "nominal": selected[0]["nominal_bernoulli_rate"],
            "observed": sum(row["observed_activity_rate"] * row["count"]
                            for row in selected) / count,
            "accepted": count,
            "attempts": sum(row["candidate_attempts"] for row in selected_events),
            "class_rejected": sum(row["class_full_rejections"]
                                  for row in selected_events),
            "duplicates": sum(row["duplicate_rejections_after_class_check"]
                              for row in selected_events),
        })
    return summary


def render_markdown(summary: list[dict], *, run_id: int) -> str:
    lines = [
        "## CL-5N — empirical synthetic data diagnostics",
        "",
        "> **POST-HOC AGGREGATE ANALYSIS ONLY.** No generation, training, "
        "inference, hardware-energy measurement, or scientific approval. "
        "H1 is **NO_GO**.",
        "",
        f"Source: [CL-5M run #{run_id}]"
        f"(https://github.com/cryptomanxxx/debatt-ai-qant-research-lab/actions/runs/{run_id}), "
        "seed 101. One run is not a population estimate.",
        "",
        "| Task | Nominal activity | Observed activity | Candidates | Accepted | Class rejects | Duplicates |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summary:
        lines.append(f"| {item['task']} | {item['nominal']:.1%} | "
                     f"{item['observed']:.1%} | {item['attempts']} | "
                     f"{item['accepted']} | {item['class_rejected']} | "
                     f"{item['duplicates']} |")
    lines += ["", "### Nominal vs observed input activity", "",
              "Each block approximates 2 percentage points; bars are descriptive, not energy measurements.", "",
              "| Task | Nominal | Observed |", "|---|---|---|"]
    for item in summary:
        left = "█" * round(50 * item["nominal"])
        right = "█" * round(50 * item["observed"])
        lines.append(f"| {item['task']} | {left} {item['nominal']:.1%} | "
                     f"{right} {item['observed']:.1%} |")
    lines += ["", "### Generator attempt overhead", "",
              "| Task | Attempts per accepted record | Rejected share |",
              "|---|---:|---:|"]
    for item in summary:
        lines.append(f"| {item['task']} | {item['attempts'] / item['accepted']:.2f} | "
                     f"{(item['attempts'] - item['accepted']) / item['attempts']:.1%} |")
    lines += ["", "Observed activity can differ from nominal Bernoulli input rates "
              "because balanced classes and unique-record acceptance condition the samples.",
              "This describes synthetic data generation, not AI architecture performance.",
              ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--run-id", required=True, type=int)
    args = parser.parse_args(argv)
    if args.run_id != 41311514598:
        parser.error("CL-5N is pinned to reviewed seed-101 source run 41311514598")
    report = json.loads(args.report.read_text(encoding="utf-8"))
    summary = render_markdown(analyze(report), run_id=args.run_id)
    with args.summary.open("a", encoding="utf-8") as handle:
        handle.write(summary)
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
