"""Five-round Dev-4 replay orchestrator: no training, API calls or fabricated outcomes.

This module is intentionally a *replay validator*, not a live benchmark. Each
round must have an externally supplied proposal and, for valid proposals, an
explicit paired evaluation or failure record. No future outcomes are passed to
a selection policy. Full GP/EI and live GPT selection are separate work.
"""
import argparse
import json
from pathlib import Path

from scripts.dev4_selection_dry_run import (
    HISTORY, PROMPT, PROTOCOL, own_round_context, validate_proposal, verify_inputs,
)

STRATEGIES = ("gpt-oss-120b", "random-search", "grid-search", "bayesian-optimization")
STATUSES = ("invalid", "evaluation_failed", "evaluated")
PAIRED_FIELDS = frozenset(("candidate_correct", "control_correct", "candidate_parameters", "control_parameters"))


def replay(protocol, records):
    """Audit five rounds per strategy without running any evaluator.

    Records are in round-major order. A valid width must have status evaluated
    or evaluation_failed. Paired scores are permitted ONLY for evaluated rows.
    """
    if not isinstance(records, list):
        raise ValueError("records must be a list")
    allowed = protocol["search_space"]["candidate_local_widths"]
    n = protocol["budget"]["selection_rounds"]
    if len(records) != n * len(STRATEGIES):
        raise ValueError("expected exactly five rounds for each of four strategies")
    ledger, outcomes, audit = [], [], []
    seen = {strategy: set() for strategy in STRATEGIES}
    for round_number in range(1, n + 1):
        for strategy in STRATEGIES:
            index = (round_number - 1) * len(STRATEGIES) + STRATEGIES.index(strategy)
            row = records[index]
            if not isinstance(row, dict) or set(row) != {"round", "strategy", "proposal", "status", "paired_outcome"}:
                raise ValueError("unexpected replay record fields")
            if row["round"] != round_number or row["strategy"] != strategy:
                raise ValueError("records must be ordered by round and strategy")
            # Capture pre-selection context only; later rounds cannot see future
            # or cross-strategy outcomes.
            context = own_round_context(strategy, ledger, outcomes)
            proposal = row["proposal"]
            if not isinstance(proposal, str):
                raise ValueError("proposal must be raw JSON text")
            width, error = validate_proposal(proposal, allowed, seen[strategy])
            # Preserve parseable invalid widths (duplicate or out of range).
            parsed_width = width
            if error is not None:
                try:
                    raw_value = json.loads(proposal)
                    if isinstance(raw_value, dict) and type(raw_value.get("local_width")) is int:
                        parsed_width = raw_value["local_width"]
                except ValueError:
                    pass
            if width is None:
                if row["status"] != "invalid" or row["paired_outcome"] is not None:
                    raise ValueError("invalid proposal cannot have an evaluation")
            else:
                if row["status"] not in ("evaluated", "evaluation_failed"):
                    raise ValueError("valid proposal requires explicit evaluation status")
                if row["status"] == "evaluation_failed":
                    if row["paired_outcome"] is not None:
                        raise ValueError("failed evaluation cannot have a fabricated paired outcome")
                else:
                    paired = row["paired_outcome"]
                    if not isinstance(paired, dict) or set(paired) != PAIRED_FIELDS:
                        raise ValueError("paired outcome must have exactly four fields")
                    if any(type(paired[k]) is not int or paired[k] < 0 for k in PAIRED_FIELDS):
                        raise ValueError("paired outcome requires nonnegative integers")
                    maximum = len(protocol["evaluation"]["seeds"]) * protocol["dataset"]["test_shape"][0]
                    if paired["candidate_correct"] > maximum or paired["control_correct"] > maximum:
                        raise ValueError("paired correct exceeds evaluation budget")
                    if paired["candidate_parameters"] == 0 or paired["control_parameters"] == 0:
                        raise ValueError("parameter counts must be positive")
                    outcomes.append({"strategy": strategy, "round": round_number, "status": "evaluated", "paired_outcome": paired})
                seen[strategy].add(width)
            ledger.append({"strategy": strategy, "round": round_number, "local_width": parsed_width,
                           "valid": error is None, "status": row["status"]})
            audit.append({"round": round_number, "strategy": strategy, "local_width": parsed_width,
                          "proposal_error": error, "status": row["status"],
                          "prior_own_attempts": len(context["proposal_status_ledger"]),
                          "prior_own_completed_outcomes": len(context["completed_paired_outcomes"])})
    return {"schema_version": 1, "mode": "externally_supplied_replay_no_compute",
            "training_runs": 0, "groq_calls": 0, "holdout_access": False,
            "rounds_per_strategy": n, "audit": audit}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-file", required=True, type=Path,
                        help="Externally supplied JSON list, NOT generated by the harness")
    args = parser.parse_args()
    protocol = json.loads(PROTOCOL.read_text())
    verify_inputs(protocol, HISTORY.read_bytes(), PROMPT.read_bytes())
    result = replay(protocol, json.loads(args.replay_file.read_text()))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
