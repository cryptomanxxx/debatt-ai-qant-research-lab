"""Dev-4 selection-only first-round dry run. Never evaluates or trains models."""
import argparse
import hashlib
import json
from pathlib import Path

from scripts.validate_dev4_selection_protocol import validate

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "research_queue/benchmarks/dev4_selection_protocol.json"
HISTORY = ROOT / "research_queue/benchmarks/dev4_initial_history_snapshot.json"
PROMPT = ROOT / "research_queue/benchmarks/dev4_gpt_oss_selection_prompt.txt"


def verify_inputs(protocol, history_bytes, prompt_bytes):
    errors = validate(protocol, ready=True)
    if errors:
        raise ValueError("invalid preregistration: " + "; ".join(errors))
    expected_history = protocol["information_policy"]["shared_initial_history_snapshot"]
    expected_prompt = protocol["strategy_specifications"]["gpt-oss-120b"]["prompt_template_sha256"]
    if "sha256:" + hashlib.sha256(history_bytes).hexdigest() != expected_history:
        raise ValueError("frozen history digest mismatch")
    if "sha256:" + hashlib.sha256(prompt_bytes).hexdigest() != expected_prompt:
        raise ValueError("GPT prompt digest mismatch")
    history = json.loads(history_bytes)
    if not history["included"] or any(
        set(obs["seeds"]) & set(protocol["evaluation"]["seeds"])
        for item in history["included"] for obs in item["observations"]
    ):
        raise ValueError("history is empty or overlaps evaluation seeds")
    return history


def own_round_context(strategy, ledger, outcomes):
    """Fail closed: expose only this policy's attempt statuses and completed outcomes."""
    allowed_status = {"invalid", "evaluation_failed", "evaluated"}
    own_ledger = []
    for row in ledger:
        if row.get("strategy") != strategy:
            continue
        if row.get("status") not in allowed_status:
            raise ValueError("unknown proposal status")
        if set(row) != {"strategy", "round", "local_width", "valid", "status"}:
            raise ValueError("proposal ledger has unexpected fields")
        own_ledger.append({k: row[k] for k in ("round", "local_width", "valid", "status")})
    own_outcomes = []
    for row in outcomes:
        if row.get("strategy") != strategy:
            continue
        if row.get("status") != "evaluated" or set(row) != {"strategy", "round", "status", "paired_outcome"}:
            raise ValueError("only completed paired outcomes may be shown")
        if not any(item["round"] == row["round"] and item["status"] == "evaluated" for item in own_ledger):
            raise ValueError("paired outcome without matching completed proposal")
        own_outcomes.append({"round": row["round"], "paired_outcome": row["paired_outcome"]})
    return {"proposal_status_ledger": own_ledger, "completed_paired_outcomes": own_outcomes}


def validate_proposal(raw, allowed, previously_proposed=()):
    """Parse one strict GPT response; invalid attempts consume their round."""
    try:
        value = json.loads(raw)
    except (TypeError, ValueError):
        return None, "invalid_json"
    if not isinstance(value, dict) or set(value) != {"local_width"}:
        return None, "invalid_schema"
    width = value["local_width"]
    if type(width) is not int or width not in allowed:
        return None, "invalid_width"
    if width in previously_proposed:
        return None, "duplicate_width"
    return width, None


def first_round(protocol, history, gpt_response=None, include_random=False):
    """Only selections, never synthetic or real paired evaluation outcomes."""
    widths = protocol["search_space"]["candidate_local_widths"]
    result = {
        "schema_version": 1,
        "mode": "selection_only_no_compute",
        "round": 1,
        "shared_history_sha256": protocol["information_policy"]["shared_initial_history_snapshot"],
        "evaluation_seeds_used": False,
        "training_runs": 0,
        "selections": {
            "grid-search": {"local_width": widths[protocol["strategy_specifications"]["grid-search"]["start_offset"]], "status": "proposed"},
            "bayesian-optimization": {"local_width": 10, "status": "fixed_initialization_no_surrogate_fit"},
            "gpt-oss-120b": {"local_width": None, "status": "awaiting_external_response_no_api_call"},
            "random-search": {"local_width": None, "status": "not_sampled_numpy_not_requested"},
        },
    }
    if gpt_response is not None:
        width, error = validate_proposal(gpt_response, widths)
        result["selections"]["gpt-oss-120b"] = {
            "local_width": width, "status": "invalid_proposal" if error else "proposed",
            "reason": error,
            "source": "supplied_response_no_api_call",
        }
    if include_random:
        try:
            import numpy as np
        except ImportError as exc:
            raise RuntimeError("NumPy PCG64 required; refusing to substitute a different RNG") from exc
        seed = protocol["strategy_specifications"]["random-search"]["seed"]
        width = int(np.random.Generator(np.random.PCG64(seed)).choice(widths, replace=False))
        result["selections"]["random-search"] = {
            "local_width": width, "status": "proposed", "numpy_version": np.__version__,
            "seed": seed,
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpt-response-file", type=Path, help="Optional existing response JSON; never calls Groq")
    parser.add_argument("--include-random", action="store_true", help="Requires NumPy PCG64")
    args = parser.parse_args()
    protocol = json.loads(PROTOCOL.read_text())
    history = verify_inputs(protocol, HISTORY.read_bytes(), PROMPT.read_bytes())
    response = args.gpt_response_file.read_text() if args.gpt_response_file else None
    result = first_round(protocol, history, response, args.include_random)
    print(json.dumps(result, indent=2, sort_keys=True))
    print("No Q.ANT training, holdout evaluation or Groq API calls performed.", flush=True)


if __name__ == "__main__":
    main()
