"""Read-only, source-pinned Research Memory v3 foundation. No network, model calls or compute.

Dev-4 visibility is strictly strategy-owned; no cross-policy result retrieval.
Only independently pinned, completed Round 1/2 feedback is indexed. Round 3
selection and future/holdout records are deliberately excluded.
"""
import hashlib
import json
from pathlib import Path

from scripts.dev4_round2_isolated_feedback import PINNED_BLOBS, git_blob_sha, verify_pinned_records

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "research_queue" / "benchmarks"
ROUNDS = ((1, "dev4_round1_verified_feedback.json"),
          (2, "dev4_round2_verified_feedback.json"))
STRATEGIES = frozenset(("gpt-oss-120b", "random-search", "grid-search",
                        "bayesian-optimization"))
DEFAULT_LIMIT = 10
MAX_LIMIT = 20
MAX_RESULT_BYTES = 12000


def _strategy(value):
    if type(value) is not str or value not in STRATEGIES:
        raise ValueError("explicit known strategy required")
    return value


def _limit(value):
    if type(value) is not int or not 1 <= value <= MAX_LIMIT:
        raise ValueError("limit outside 1..20")
    return value


def _records(strategy):
    strategy = _strategy(strategy)
    verify_pinned_records()
    records = []
    for number, filename in ROUNDS:
        raw = (BENCH / filename).read_bytes()
        if git_blob_sha(raw) != PINNED_BLOBS[filename]:
            raise ValueError("historical source changed")
        source = json.loads(raw)
        row = source["strategies"][strategy]
        control = source["control"]
        if source["schema_version"] != 1 or source["source"]["backend"] != "qant-cpu/software-simulation":
            raise ValueError("unsupported source")
        if source["evaluation"]["seeds"] != [301, 302, 303, 304, 305]:
            raise ValueError("evaluation protocol mismatch")
        if type(row["local_width"]) is not int or type(row["candidate_correct"]) is not int:
            raise ValueError("invalid result")
        records.append({
            "experiment_id": f"dev4-r{number}-{strategy}",
            "round": number,
            "strategy": strategy,
            "dataset": "ECG200",
            "local_width": row["local_width"],
            "epochs": source["evaluation"]["epochs"],
            "seeds": list(source["evaluation"]["seeds"]),
            "candidate_correct": row["candidate_correct"],
            "control_correct": control["candidate_correct"],
            "candidate_parameters": row["candidate_parameters"],
            "control_parameters": control["candidate_parameters"],
            "gate_pass": row["candidate_correct"] >= control["candidate_correct"] - source["evaluation"]["gate_margin_correct"],
            "source": {
                "path": "research_queue/benchmarks/" + filename,
                "git_blob_sha1": PINNED_BLOBS[filename],
                "original_artifact_sha256": source["source"]["artifact_sha256"],
                "workflow_run_id": source["source"]["workflow_run_id"],
            },
        })
    return tuple(records)


def _bounded(records, limit):
    data = {"schema_version": 1, "read_only": True, "count": len(records),
            "results": list(records[:_limit(limit)]), "truncated": len(records) > limit}
    if len(json.dumps(data, sort_keys=True).encode("utf-8")) > MAX_RESULT_BYTES:
        raise ValueError("retrieval output exceeds byte budget")
    return data


def search_experiments(*, strategy, dataset="ECG200", local_width=None,
                       epochs=None, limit=DEFAULT_LIMIT):
    """Exact-match query of completed, own-strategy Dev-4 results only."""
    _strategy(strategy)
    _limit(limit)
    if dataset != "ECG200":
        raise ValueError("unsupported dataset")
    if local_width is not None and (type(local_width) is not int or not 4 <= local_width <= 15):
        raise ValueError("invalid width filter")
    if epochs is not None and (type(epochs) is not int or epochs != 100):
        raise ValueError("unsupported epochs")
    records = [row for row in _records(strategy)
               if (local_width is None or row["local_width"] == local_width)
               and (epochs is None or row["epochs"] == epochs)]
    return _bounded(records, limit)


def get_experiment(*, strategy, experiment_id):
    """An ID from another strategy is never readable, even when guessed."""
    _strategy(strategy)
    if type(experiment_id) is not str or len(experiment_id) > 100:
        raise ValueError("invalid experiment ID")
    records = [row for row in _records(strategy) if row["experiment_id"] == experiment_id]
    if len(records) != 1:
        raise ValueError("experiment not found in authorized own history")
    return _bounded(records, 1)


def get_pareto_front(*, strategy, limit=DEFAULT_LIMIT):
    """Own-only non-dominated records: maximize correct, minimize parameters."""
    records = _records(strategy)
    front = [row for row in records if not any(
        (other["candidate_correct"] >= row["candidate_correct"]
         and other["candidate_parameters"] <= row["candidate_parameters"])
        and (other["candidate_correct"] > row["candidate_correct"]
             or other["candidate_parameters"] < row["candidate_parameters"])
        for other in records)]
    return _bounded(front, limit)


def demonstration():
    """Static, deterministic offline demonstration; never calls an external service."""
    return {"search": search_experiments(strategy="gpt-oss-120b"),
            "experiment": get_experiment(strategy="gpt-oss-120b",
                                         experiment_id="dev4-r2-gpt-oss-120b"),
            "pareto": get_pareto_front(strategy="gpt-oss-120b"),
            "groq_calls": 0, "training_runs": 0}


if __name__ == "__main__":
    print(json.dumps(demonstration(), sort_keys=True, indent=2))
