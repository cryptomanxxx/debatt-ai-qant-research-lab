"""Build round-two strategy-isolated feedback from verified Dev-4 round-one evidence.

Selection context only: no Groq call, Q.ANT compute, holdout reads, or round-two
proposals. The immutable GitHub artifact is authoritative; the committed
feedback record is checked against it when --source-artifact is supplied.
"""
import argparse
import hashlib
import json
from pathlib import Path

from scripts.dev4_selection_dry_run import own_round_context
from scripts.validate_dev4_selection_protocol import validate

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "research_queue/benchmarks/dev4_selection_protocol.json"
FEEDBACK = ROOT / "research_queue/benchmarks/dev4_round1_verified_feedback.json"


def verified_feedback(protocol, feedback, source_bytes=None):
    if validate(protocol, ready=True):
        raise ValueError("preregistration invalid")
    source = feedback["source"]
    if source["workflow_run_id"] != 36559946601 or source["artifact_id"] != 11029104019:
        raise ValueError("unexpected source run/artifact")
    if source["source_selection_run"] != 36555941972 or source["source_response_id"] != "chatcmpl-b47f2f0a-e9f8-4864-bb80-2a288726a91f":
        raise ValueError("unexpected original Groq selection")
    if source["backend"] != "qant-cpu/software-simulation" or source["training_runs"] != 25:
        raise ValueError("unexpected backend or compute count")
    evaluation = protocol["evaluation"]
    for key in ("seeds", "epochs", "learning_rate", "batch_size", "frequencies", "gate_margin_correct"):
        if feedback["evaluation"][key] != evaluation[key]:
            raise ValueError("protocol mismatch: " + key)
    if feedback["evaluation"]["control_local_width"] != protocol["search_space"]["control_local_width"]:
        raise ValueError("control width mismatch")
    if set(feedback["strategies"]) != set(protocol["strategies"]):
        raise ValueError("strategy set mismatch")
    if feedback["control"] != {"candidate_correct": 452, "candidate_parameters": 8550}:
        raise ValueError("control mismatch")
    if source_bytes is not None:
        if hashlib.sha256(source_bytes).hexdigest() != source["artifact_sha256"]:
            raise ValueError("original artifact SHA256 mismatch")
        actual = json.loads(source_bytes)
        if (actual["training_runs"] != 25 or actual["source_selection_run"] != source["source_selection_run"]
                or actual["source_response_id"] != source["source_response_id"]
                or actual["backend"] != source["backend"]):
            raise ValueError("original artifact metadata mismatch")
        if actual["configuration"]["seeds"] != evaluation["seeds"] or actual["configuration"]["selections"] != {
                name: row["local_width"] for name, row in feedback["strategies"].items()}:
            raise ValueError("original selections mismatch")
        # The authoritative artifact must itself match the preregistration,
        # not merely agree with the committed feedback record.
        if actual["dataset"] != protocol["dataset"]:
            raise ValueError("original dataset mismatch")
        expected_config = {key: evaluation[key] for key in
                           ("seeds", "epochs", "learning_rate", "batch_size",
                            "frequencies", "gate_margin_correct")}
        expected_config["control_local_width"] = protocol["search_space"]["control_local_width"]
        expected_config["selections"] = {
            name: row["local_width"] for name, row in feedback["strategies"].items()}
        if actual["configuration"] != expected_config:
            raise ValueError("original full evaluation configuration mismatch")
        if actual["control_reused_across_strategies"] is not True:
            raise ValueError("shared control declaration mismatch")
        expected_names = set(protocol["strategies"]) | {"control"}
        expected_pairs = {(name, seed) for name in expected_names for seed in evaluation["seeds"]}
        rows = actual["rows"]
        pairs = [(row["candidate"], row["seed"]) for row in rows]
        if len(rows) != len(expected_pairs) or len(set(pairs)) != len(pairs) or set(pairs) != expected_pairs:
            raise ValueError("incomplete or duplicate original candidate/seed rows")
        if set(actual["summary"]) != expected_names:
            raise ValueError("original summary candidate set mismatch")
        test_size = protocol["dataset"]["test_shape"][0]
        for name in expected_names:
            group = [row for row in rows if row["candidate"] == name]
            expected_width = (expected_config["control_local_width"] if name == "control"
                              else expected_config["selections"][name])
            parameters = {row["parameter_count"] for row in group}
            if (any(row["local_width"] != expected_width for row in group)
                    or len(parameters) != 1 or next(iter(parameters)) <= 0):
                raise ValueError("original row width/parameter mismatch: " + name)
            for row in group:
                if (any(type(row[key]) is not int or not 0 <= row[key] <= test_size
                        for key in ("qant_correct", "reference_correct"))
                        or type(row["prediction_disagreements"]) is not int
                        or not 0 <= row["prediction_disagreements"] <= test_size):
                    raise ValueError("invalid original per-seed measurement: " + name)
            summary = actual["summary"][name]
            qant = sum(row["qant_correct"] for row in group)
            reference = sum(row["reference_correct"] for row in group)
            if (summary["local_width"] != expected_width
                    or summary["parameter_count"] != next(iter(parameters))
                    or summary["aggregate_qant_correct"] != qant
                    or summary["aggregate_reference_correct"] != reference):
                raise ValueError("original summary disagrees with seed rows: " + name)
            expected = (feedback["control"] if name == "control"
                        else feedback["strategies"][name])
            if (qant, summary["parameter_count"]) != (
                    expected["candidate_correct"], expected["candidate_parameters"]):
                raise ValueError("original measured result mismatch: " + name)
            if name != "control":
                gate = qant >= feedback["control"]["candidate_correct"] - evaluation["gate_margin_correct"]
                saving = 1 - summary["parameter_count"] / feedback["control"]["candidate_parameters"]
                if summary["gate_pass"] is not gate or abs(summary["parameter_saving"] - saving) > 1e-12:
                    raise ValueError("original gate/parameter saving mismatch: " + name)
    return feedback


def build_contexts(protocol, feedback):
    verified_feedback(protocol, feedback)
    ledgers, outcomes = [], []
    control = feedback["control"]
    for strategy in protocol["strategies"]:
        row = feedback["strategies"][strategy]
        width = row["local_width"]
        if type(width) is not int or width not in protocol["search_space"]["candidate_local_widths"]:
            raise ValueError("width outside preregistered space")
        paired = {"candidate_correct": row["candidate_correct"],
                  "control_correct": control["candidate_correct"],
                  "candidate_parameters": row["candidate_parameters"],
                  "control_parameters": control["candidate_parameters"]}
        if any(type(v) is not int or v <= 0 for v in paired.values()):
            raise ValueError("invalid measured paired outcome")
        if paired["candidate_correct"] > len(protocol["evaluation"]["seeds"]) * protocol["dataset"]["test_shape"][0]:
            raise ValueError("accuracy exceeds test size")
        ledgers.append({"strategy": strategy, "round": 1, "local_width": width,
                        "valid": True, "status": "evaluated"})
        outcomes.append({"strategy": strategy, "round": 1, "status": "evaluated", "paired_outcome": paired})
    return {
        "schema_version": 1, "mode": "round_two_feedback_only_no_selection_no_compute",
        "source": feedback["source"],
        "shared_initial_history_sha256": protocol["information_policy"]["shared_initial_history_snapshot"],
        "contexts": {strategy: own_round_context(strategy, ledgers, outcomes)
                     for strategy in protocol["strategies"]},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-artifact", type=Path, required=True,
                        help="Original dev4_round1_qant_cpu.json downloaded from pinned GitHub run")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(PROTOCOL.read_text())
    feedback = json.loads(FEEDBACK.read_text())
    verified_feedback(protocol, feedback, args.source_artifact.read_bytes())
    result = build_contexts(protocol, feedback)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("Verified four isolated round-two contexts; no selections, Groq calls or training.")


if __name__ == "__main__":
    main()
