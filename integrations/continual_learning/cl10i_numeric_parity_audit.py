"""CL-10I pure numerical parity audit for *supplied* future trace records.

No model imports, training, inference, datasets or physical measurements.
Candidate tolerances are NOT scientifically preapproved and never grant GO.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

SPEC_PATH = Path(__file__).with_name("cl10i_numeric_parity_candidate.json")
PAIRS = {
    "primary_sparse": ("local_hybrid", "sparse_local_hybrid"),
    "secondary_index_reuse": ("sparse_local_hybrid", "sparse_shared_index"),
}
NUMERIC = ("pre_weights", "hidden", "logits", "loss", "post_weights")
LENGTHS = {"pre_weights": 154, "hidden": 8, "logits": 2, "post_weights": 154}


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError("CL-10I NO_GO: " + reason)


def validate_spec(spec: dict) -> None:
    _require(type(spec) is dict
             and spec.get("kind") == "CL10I_PAIRED_NUMERIC_PARITY_CANDIDATE"
             and spec.get("decision") == "NO_GO"
             and spec.get("review_only") is True
             and spec.get("training_authorized") is False
             and spec.get("inference_authorized") is False
             and spec.get("dataset_generation_authorized") is False
             and spec.get("observations_recorded") == 0,
             "execution guard changed")
    _require(spec.get("pairs") == {k: list(v) for k, v in PAIRS.items()},
             "causal pair definitions changed")
    tol = spec.get("candidate_tolerance", {})
    _require(tol.get("absolute") == 1e-12
             and tol.get("relative") == 1e-10
             and tol.get("method") == "abs_a_minus_b_le_max_abs_or_relative"
             and tol.get("preapproved") is False,
             "proposed numeric tolerance changed or falsely approved")
    required = spec.get("required_trace", {})
    _require(required.get("parameter_count") == 154
             and required.get("hidden_count") == 8
             and required.get("logit_count") == 2
             and required.get("active_indices_order") == "sorted_unique"
             and required.get("source_and_sample_identity_required") is True
             and required.get("pre_and_post_state_required") is True
             and required.get("loss_required") is True
             and required.get("optimizer_step_must_match") is True,
             "trace scope changed")
    limits = spec.get("limitations", {})
    _require(limits.get("independent_scientific_signoff") is False
             and limits.get("real_trace_authenticity_verified") is False
             and limits.get("actual_parity_verified") is False
             and limits.get("physical_energy") == "NOT_MEASURED",
             "false scientific readiness claim")


def _vector(value: object, count: int) -> list[float]:
    _require(type(value) is list and len(value) == count, "numeric vector shape")
    _require(all(type(x) in (int, float) and math.isfinite(x) for x in value),
             "nonfinite/bool numeric vector")
    return value


def _arm_trace(trace: object, arm: str) -> None:
    _require(type(trace) is dict and set(trace) ==
             set(NUMERIC) | {"arm", "active_indices", "optimizer_steps"},
             "trace fields missing/extra")
    _require(trace["arm"] == arm, "unexpected arm")
    for field, length in LENGTHS.items():
        _vector(trace[field], length)
    _require(type(trace["loss"]) in (int, float)
             and math.isfinite(trace["loss"]), "invalid loss")
    indices = trace["active_indices"]
    _require(type(indices) is list
             and all(type(i) is int and 0 <= i < 16 for i in indices)
             and indices == sorted(set(indices)), "active indices invalid")
    _require(type(trace["optimizer_steps"]) is int
             and 1 <= trace["optimizer_steps"] <= 512,
             "optimizer steps invalid")


def audit_supplied_trace(spec: dict, record: object) -> dict:
    """Structural parity only. A matching fabricated trace is NOT evidence."""
    validate_spec(spec)
    _require(type(record) is dict and set(record) ==
             {"pair_id", "seed", "task", "update_index", "source_sha",
              "sample_sha256", "left", "right"}, "record fields invalid")
    pair = record["pair_id"]
    _require(type(pair) is str and pair in PAIRS, "invalid causal pair")
    _require(type(record["seed"]) is int
             and record["seed"] in (101, 102, 103, 104, 105)
             and record["task"] in ("T1", "T2", "T3", "T4")
             and type(record["update_index"]) is int
             and 1 <= record["update_index"] <= 512,
             "record identity invalid")
    _require(type(record["source_sha"]) is str
             and re.fullmatch(r"[0-9a-f]{40}", record["source_sha"]) is not None
             and type(record["sample_sha256"]) is str
             and re.fullmatch(r"[0-9a-f]{64}", record["sample_sha256"]) is not None,
             "source/sample digest invalid")
    left_name, right_name = PAIRS[pair]
    left, right = record["left"], record["right"]
    _arm_trace(left, left_name)
    _arm_trace(right, right_name)
    _require(left["optimizer_steps"] == right["optimizer_steps"]
             == record["update_index"], "optimizer count mismatch")
    _require(left["active_indices"] == right["active_indices"],
             "active set mismatch")
    tol = spec["candidate_tolerance"]
    first_mismatch = None
    checked = 0
    max_abs_difference = 0.0
    for field in NUMERIC:
        a_values = [left["loss"]] if field == "loss" else left[field]
        b_values = [right["loss"]] if field == "loss" else right[field]
        for i, (a, b) in enumerate(zip(a_values, b_values)):
            difference = abs(a - b)
            _require(math.isfinite(difference), "numeric difference overflow")
            max_abs_difference = max(max_abs_difference, difference)
            checked += 1
            if (first_mismatch is None and
                    not math.isclose(a, b, rel_tol=tol["relative"],
                                     abs_tol=tol["absolute"])):
                first_mismatch = f"{field}[{i}]"
    return {
        "decision": "NO_GO",
        "scope": "UNAUTHENTICATED_SUPPLIED_TRACE_ONLY",
        "candidate_parity_within_tolerance": first_mismatch is None,
        "first_mismatch": first_mismatch,
        "compared_numeric_values": checked,
        "max_absolute_difference": max_abs_difference,
        "observations_authenticated": False,
        "tolerance_preapproved": False,
        "training_authorized": False,
    }


def main() -> None:
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    validate_spec(spec)
    print(json.dumps({"decision": "NO_GO", "scope": "STATIC_REVIEW_ONLY",
                      "models_constructed": 0, "observations_recorded": 0}))


if __name__ == "__main__":
    main()
