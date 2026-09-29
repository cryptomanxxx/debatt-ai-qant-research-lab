"""Validate Dev-4 preregistration only; this module never selects or executes jobs."""
import argparse
import json
from pathlib import Path

EXPECTED = ("gpt-oss-120b", "random-search", "grid-search", "bayesian-optimization")
METRIC = "number_of_valid_gate_passing_pareto_candidates_within_budget"


def pinned_digest(value):
    return (isinstance(value, str) and len(value) == 71 and value.startswith("sha256:")
            and all(c in "0123456789abcdef" for c in value[7:]))


def validate(p, ready=False):
    errors = []
    if not isinstance(p, dict):
        return ["protocol must be an object"]
    if p.get("schema_version") != 1 or p.get("status") != "preregistered_design_only":
        errors.append("protocol version/status mismatch")
    strategies = p.get("strategies")
    if not isinstance(strategies, list) or len(strategies) != 4 or set(strategies) != set(EXPECTED):
        errors.append("four unique preregistered strategies required")
    dataset = p.get("dataset", {})
    if not isinstance(dataset, dict) or dataset != {
        "name": "ECG200", "train_shape": [100, 96], "test_shape": [100, 96]
    }:
        errors.append("dataset shape mismatch")
    space = p.get("search_space", {})
    if not isinstance(space, dict) or space.get("candidate_local_widths") != list(range(4, 16)) or space.get("control_local_width") != 16:
        errors.append("search space mismatch")
    evaluation = p.get("evaluation", {})
    if not isinstance(evaluation, dict) or evaluation.get("epochs") != 100 or evaluation.get("gate_margin_correct") != 10:
        errors.append("evaluation contract mismatch")
    else:
        seeds = evaluation.get("seeds")
        if (not isinstance(seeds, list) or len(seeds) != 5
            or any(type(x) is not int or x <= 0 for x in seeds) or len(set(seeds)) != 5
            or type(evaluation.get("learning_rate")) not in (int, float)
            or not 0 < evaluation["learning_rate"] < float("inf")
            or type(evaluation.get("batch_size")) is not int or evaluation["batch_size"] <= 0
            or not isinstance(evaluation.get("frequencies"), list)
            or not evaluation["frequencies"]
            or any(type(x) is not int or x <= 0 for x in evaluation["frequencies"])):
            errors.append("invalid paired evaluation protocol")
    budget = p.get("budget", {})
    if not isinstance(budget, dict):
        errors.append("budget missing")
    else:
        rounds = budget.get("selection_rounds")
        proposals = budget.get("proposals_per_round")
        candidates = budget.get("max_evaluated_candidates_per_strategy")
        runs = budget.get("max_training_runs_per_strategy")
        if (type(rounds) is not int or type(proposals) is not int
            or type(candidates) is not int or type(runs) is not int
            or (rounds, proposals, candidates, runs) != (5, 1, 5, 50)):
            errors.append("preregistered budget must be exactly 5/1/5/50")
    info = p.get("information_policy", {})
    if (not isinstance(info, dict)
        or info.get("history_visibility") != "identical initial snapshot and only own prior evaluated rounds"
        or info.get("candidate_feedback") != "same structured paired outcome schema for every strategy"
        or info.get("holdout_policy") != "no access to evaluation results before selection"):
        errors.append("unequal or unspecified information policy")
    elif ready and not pinned_digest(info.get("shared_initial_history_snapshot")):
        errors.append("execution blocked: initial history must be pinned by SHA-256")
    if ready:
        specs = p.get("strategy_specifications")
        required = {
            "random-search": ("rng_algorithm", "seed", "sampling_policy"),
            "grid-search": ("traversal_order", "start_offset"),
            "bayesian-optimization": ("surrogate", "acquisition", "initialization", "random_seed", "optimizer_policy"),
            "gpt-oss-120b": ("model_identifier", "prompt_template_sha256", "decoding_policy", "context_policy"),
        }
        if not isinstance(specs, dict) or set(specs) != set(EXPECTED):
            errors.append("execution blocked: all four strategy specifications required")
        else:
            for name, keys in required.items():
                spec = specs[name]
                if not isinstance(spec, dict) or any(
                    key not in spec or spec[key] is None or spec[key] == "" or spec[key] == "TO_BE_DEFINED"
                    for key in keys
                ):
                    errors.append("execution blocked: incomplete strategy specification: " + name)
            if isinstance(specs.get("random-search"), dict) and type(specs["random-search"].get("seed")) is not int:
                errors.append("random-search seed must be an integer")
            if isinstance(specs.get("bayesian-optimization"), dict) and type(specs["bayesian-optimization"].get("random_seed")) is not int:
                errors.append("bayesian random seed must be an integer")
            if isinstance(specs.get("gpt-oss-120b"), dict) and not pinned_digest(specs["gpt-oss-120b"].get("prompt_template_sha256")):
                errors.append("GPT prompt template must be pinned by SHA-256")
    if p.get("primary_metric") != METRIC:
        errors.append("primary metric changed")
    guard = p.get("guardrails", {})
    if not isinstance(guard, dict) or guard.get("human_approval_required") is not True or guard.get("automatic_compute") is not False or guard.get("synthetic_results_are_not_scientific_evidence") is not True:
        errors.append("mandatory safety guardrails missing")
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=Path("research_queue/benchmarks/dev4_selection_protocol.json"))
    parser.add_argument("--ready-for-execution", action="store_true", help="Require pinned history and complete strategy specifications (does not authorize compute)")
    args = parser.parse_args()
    errors = validate(json.loads(args.protocol.read_text()), ready=args.ready_for_execution)
    if errors:
        parser.exit(1, "\n".join(errors) + "\n")
    print("Dev-4 protocol valid (validation does not authorize compute).")


if __name__ == "__main__":
    main()
