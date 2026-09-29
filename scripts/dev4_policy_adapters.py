"""Deterministic Dev-4 selection adapters. No training, holdout reads or API calls.

Only a strategy's own prior completed paired outcomes may inform Bayesian
optimization. GPT responses are externally supplied; this module never calls Groq.
"""
import math

from scripts.dev4_selection_dry_run import validate_proposal


def choose_grid(protocol, round_number):
    widths = protocol["search_space"]["candidate_local_widths"]
    offset = protocol["strategy_specifications"]["grid-search"]["start_offset"]
    if type(round_number) is not int or not 1 <= round_number <= protocol["budget"]["selection_rounds"]:
        raise ValueError("round outside preregistered budget")
    return widths[offset + round_number - 1]


def random_stream(protocol):
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("NumPy PCG64 required; no RNG substitution") from exc
    widths = protocol["search_space"]["candidate_local_widths"]
    seed = protocol["strategy_specifications"]["random-search"]["seed"]
    n = protocol["budget"]["selection_rounds"]
    return [int(x) for x in np.random.Generator(np.random.PCG64(seed)).choice(
        widths, size=n, replace=False
    )]


def choose_random(protocol, round_number):
    if type(round_number) is not int or not 1 <= round_number <= protocol["budget"]["selection_rounds"]:
        raise ValueError("round outside preregistered budget")
    return random_stream(protocol)[round_number - 1]


def choose_bayesian(protocol, own_ledger, own_completed_outcomes):
    """Discrete GP/Matern(2.5) + EI(xi=.01), using own prior valid outcomes only.

    Both inputs must already be isolated to Bayesian optimization. A ledger
    entry is {round,local_width,valid,status}; a completed outcome entry is
    {round,paired_outcome}. Never fit to failed/invalid proposals.
    """
    import numpy as np
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern

    widths = protocol["search_space"]["candidate_local_widths"]
    if len(own_ledger) >= protocol["budget"]["selection_rounds"]:
        raise ValueError("selection budget exhausted")
    if any(set(row) != {"round", "local_width", "valid", "status"} for row in own_ledger):
        raise ValueError("invalid own proposal ledger")
    rounds = [row["round"] for row in own_ledger]
    if rounds != list(range(1, len(rounds) + 1)):
        raise ValueError("own proposal ledger is not chronological")
    proposed = {row["local_width"] for row in own_ledger if type(row["local_width"]) is int}
    remaining = [width for width in widths if width not in proposed]
    if not remaining:
        raise ValueError("no remaining widths")
    if not own_ledger:
        return 10
    evaluated = {row["round"]: row for row in own_ledger if row["status"] == "evaluated" and row["valid"] is True}
    if len(own_completed_outcomes) != len(evaluated):
        raise ValueError("completed outcome ledger mismatch")
    observations = []
    for item in own_completed_outcomes:
        if set(item) != {"round", "paired_outcome"} or item["round"] not in evaluated:
            raise ValueError("unexpected completed outcome")
        paired = item["paired_outcome"]
        if not isinstance(paired, dict) or set(paired) != {
            "candidate_correct", "control_correct", "candidate_parameters", "control_parameters"
        } or any(type(v) is not int or v < 0 for v in paired.values()):
            raise ValueError("invalid paired outcome")
        width = evaluated[item["round"]]["local_width"]
        if width not in widths:
            raise ValueError("outcome width outside search space")
        observations.append((width, paired["candidate_correct"] - paired["control_correct"],
                             paired["control_parameters"] - paired["candidate_parameters"]))
    if not observations:
        return remaining[0]
    x = np.asarray([[row[0]] for row in observations], dtype=float)
    y = np.asarray([row[1] for row in observations], dtype=float)
    model = GaussianProcessRegressor(
        kernel=Matern(nu=2.5), normalize_y=True, alpha=1e-6,
        random_state=protocol["strategy_specifications"]["bayesian-optimization"]["random_seed"],
    )
    model.fit(x, y)
    means, stds = model.predict(np.asarray(remaining, dtype=float).reshape(-1, 1), return_std=True)
    best = max(y)
    xi = 0.01
    scores = []
    for width, mean, std in zip(remaining, means, stds):
        improvement = float(mean) - best - xi
        if std <= 0:
            ei = max(0.0, improvement)
        else:
            z = improvement / float(std)
            cdf = 0.5 * (1 + math.erf(z / math.sqrt(2)))
            pdf = math.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)
            ei = improvement * cdf + float(std) * pdf
        # EI ties: deterministic lower-width choice. Parameter saving cannot
        # be known for an unevaluated width without a preregistered estimator.
        scores.append((ei, -width, width))
    return max(scores)[2]


def choose_gpt_external(protocol, response, own_ledger):
    """Parse supplied GPT JSON only; no network or retry."""
    attempted = {row["local_width"] for row in own_ledger if type(row["local_width"]) is int}
    return validate_proposal(response, protocol["search_space"]["candidate_local_widths"], attempted)
