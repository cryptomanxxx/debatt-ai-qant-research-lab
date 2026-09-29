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
        if len(actual["rows"]) != 25:
            raise ValueError("incomplete original seed rows")
        for name, row in feedback["strategies"].items():
            summary = actual["summary"][name]
            if (summary["local_width"], summary["aggregate_qant_correct"], summary["parameter_count"]) != (
                    row["local_width"], row["candidate_correct"], row["candidate_parameters"]):
                raise ValueError("original measured result mismatch: " + name)
        control = actual["summary"]["control"]
        if (control["aggregate_qant_correct"], control["parameter_count"]) != (
                feedback["control"]["candidate_correct"], feedback["control"]["candidate_parameters"]):
            raise ValueError("original control result mismatch")
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
