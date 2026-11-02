"""CL-10H: static train-only learnability score contract. NO_GO.

No models, datasets, inference or training are constructed. This routine only
checks externally supplied count-shaped records against the preregistered
candidate; it cannot authenticate observations or approve scientific claims.
"""
from __future__ import annotations

import json
from pathlib import Path

BASE = Path(__file__).resolve().parent
CANDIDATE = BASE / "cl10h_train_only_control_candidate.json"
SEEDS = (101, 102, 103, 104, 105)
TASKS = ("T1", "T2", "T3", "T4")


def _require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError("CL-10H NO_GO: " + reason)


def validate_candidate(spec: dict) -> None:
    _require(type(spec) is dict and
             spec.get("kind") == "CL10H_TRAIN_ONLY_LEARNABILITY_CONTROL_CANDIDATE"
             and spec.get("decision") == "NO_GO"
             and spec.get("review_only") is True
             and spec.get("training_authorized") is False
             and spec.get("inference_authorized") is False
             and spec.get("dataset_generation_authorized") is False
             and spec.get("observations_recorded") == 0,
             "review-only control guard changed")
    control = spec.get("control", {})
    _require(control.get("class") ==
             "integrations.continual_learning.cl2_mlp_baseline.TinyDenseMLP"
             and control.get("architecture") == [16, 8, 2]
             and control.get("parameters") == 154
             and control.get("seeds") == list(SEEDS)
             and control.get("tasks") == list(TASKS)
             and control.get("calibration_split_algorithm") ==
             "cl10g-train-only-stratified-hash-v1"
             and control.get("fit_per_task") == 96
             and control.get("validation_per_task") == 32
             and control.get("fit_updates_per_task") == 96
             and control.get("passes") == 1
             and control.get("total_fit_updates_per_seed") == 384
             and control.get("learning_rate") == 0.05
             and control.get("hyperparameter_tuning") is False
             and control.get("heldout_access") is False
             and control.get("checkpoint") == "post_T4",
             "model/fit/test scope drift")
    gate = spec.get("candidate_decision", {})
    _require(gate.get("per_seed_validation_examples") == 128
             and gate.get("validation_examples_per_task") == 32
             and gate.get("minimum_total_correct_per_seed") == 84
             and gate.get("accuracy_floor") == 0.65
             and gate.get("required_passing_seeds") == 5
             and gate.get("majority_baseline_accuracy") == 0.5
             and gate.get("if_any_seed_fails") == "STOP_NO_LEARNABILITY_CLAIM"
             and gate.get("if_any_record_missing") == "STOP_REPORT_ALL"
             and gate.get("threshold_frozen_before_data_access") is True,
             "candidate decision rule drift")
    limits = spec.get("limitations", {})
    _require(limits.get("independent_scientific_signoff") is False
             and limits.get("learnability_verified") is False
             and limits.get("source_data_frozen") is False
             and limits.get("numerical_tolerance_approved") is False
             and limits.get("implementation_not_authorized") is True
             and limits.get("physical_energy") == "NOT_MEASURED",
             "missing unresolved scientific gates")


def assess_supplied_counts(spec: dict, records: object) -> dict:
    """Structural score arithmetic only; never declare a successful experiment."""
    validate_candidate(spec)
    _require(type(records) is list and len(records) == 20,
             "expected five seeds times four tasks")
    index = {}
    for row in records:
        _require(type(row) is dict and set(row) ==
                 {"seed", "task", "correct", "eligible", "source"},
                 "unexpected row fields")
        seed, task = row["seed"], row["task"]
        _require(type(seed) is int and seed in SEEDS and type(task) is str and task in TASKS
                 and row["source"] == "SUPPLIED_UNVERIFIED"
                 and type(row["correct"]) is int
                 and type(row["eligible"]) is int
                 and row["eligible"] == 32
                 and 0 <= row["correct"] <= 32,
                 "invalid record identity, source or count")
        key = (seed, task)
        _require(key not in index, "duplicate seed/task record")
        index[key] = row["correct"]
    _require(set(index) == {(s, t) for s in SEEDS for t in TASKS},
             "incomplete calibration validation matrix")
    results = [
        {"seed": seed, "correct": sum(index[seed, task] for task in TASKS),
         "eligible": 128,
         "meets_candidate_floor": sum(index[seed, task] for task in TASKS) >= 84}
        for seed in SEEDS
    ]
    return {
        "decision": "NO_GO", "scope": "SUPPLIED_COUNTS_ARITHMETIC_ONLY",
        "candidate_all_seed_floor_met": all(x["meets_candidate_floor"] for x in results),
        "per_seed": results, "observations_authenticated": False,
        "learnability_verified": False, "training_authorized": False,
    }


def main() -> None:
    # CI reads the proposal only. No record generator, model or dataset imported.
    spec = json.loads(CANDIDATE.read_text(encoding="utf-8"))
    validate_candidate(spec)
    print(json.dumps({"decision": "NO_GO", "scope": "STATIC_REVIEW_ONLY",
                      "models_constructed": 0, "observations_recorded": 0}))


if __name__ == "__main__":
    main()
