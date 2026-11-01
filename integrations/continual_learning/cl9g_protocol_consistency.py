"""CL-9G fail-closed static H1 protocol reconciliation. No model imports or execution."""
from __future__ import annotations

import json
from pathlib import Path

BASE = Path(__file__).resolve().parent
EXPECTED_SEEDS = [101, 102, 103, 104, 105]
EXPECTED_ARMS = [
    "dense_sgd", "local_hybrid", "sparse_local_hybrid",
    "sparse_shared_index",
]
EXPECTED_CAUSAL = [
    ("primary_sparse", "local_hybrid", "sparse_local_hybrid"),
    ("secondary_index_reuse", "sparse_local_hybrid", "sparse_shared_index"),
]


def _read(name: str) -> dict:
    data = json.loads((BASE / name).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Invalid protocol document: {name}")
    return data


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(f"CL-9G NO_GO: {message}")


def reconcile() -> dict:
    """Validate exact static source-contract agreement, never certify readiness."""
    source = _read("cl6a_readiness_protocol.json")
    readiness = _read("cl9a_minimal_h1_readiness.json")
    acceptance = _read("cl9b_h1_acceptance_contract.json")
    measurement = _read("cl9c_measurement_protocol.json")
    calibration = _read("cl7c_train_only_calibration_protocol.json")
    parity = _read("cl7d_causal_pairing_invariants.json")
    reproducibility = _read("cl9d_reproducibility_failure_policy.json")
    critique = _read("cl7a_scientific_design_critique.json")

    for name, obj in (
        ("CL-6A", source), ("CL-9A", readiness), ("CL-9B", acceptance),
        ("CL-9C", measurement), ("CL-7C", calibration),
        ("CL-7D", parity), ("CL-9D", reproducibility),
        ("CL-7A", critique),
    ):
        _require(obj.get("decision") == "NO_GO", f"{name} decision changed")
    denied_flags = (
        (source, "model_training_authorized"),
        (source, "model_inference_authorized"),
        (source, "dataset_generation_authorized"),
        (readiness, "model_construction_authorized"),
        (readiness, "training_authorized"),
        (readiness, "model_inference_authorized"),
        (readiness, "dataset_generation_authorized"),
        (acceptance, "model_training_authorized"),
        (acceptance, "model_inference_authorized"),
        (acceptance, "dataset_generation_authorized"),
        (measurement, "model_training_authorized"),
        (measurement, "model_inference_authorized"),
        (measurement, "synthetic_data_generation_authorized"),
        (calibration, "training_authorized"),
        (calibration, "model_inference_authorized"),
        (calibration, "dataset_generation_authorized"),
        (parity, "training_authorized"),
        (parity, "model_inference_authorized"),
        (parity, "dataset_generation_authorized"),
        (reproducibility, "training_authorized"),
        (reproducibility, "inference_authorized"),
        (reproducibility, "new_data_generation_authorized"),
    )
    for obj, field in denied_flags:
        _require(obj.get(field) is False, f"{obj.get('kind', 'contract')}.{field} must be false")

    arms = [item["id"] for item in source["arms"]]
    seeds = source["future_design"]["seeds"]
    causal = [
        (pair["id"], pair["left"], pair["right"])
        for pair in source["comparisons"] if pair["id"] != "context_learning_rule"
    ]
    acceptance_causal = [
        (pair["id"], pair["reference"], pair["candidate"])
        for pair in acceptance["contrasts"]
    ]
    parity_causal = [
        (pair["id"], pair["left"], pair["right"])
        for pair in parity["causal_pairs"]
    ]
    _require(source["model"]["shape"] == [16, 8, 2], "unexpected model shape")
    _require(arms == EXPECTED_ARMS, "arms changed or duplicated")
    _require(seeds == EXPECTED_SEEDS, "source seed list changed")
    _require(acceptance["descriptive_seeds"] == seeds, "acceptance seeds differ")
    _require(measurement["paired_design"]["seeds"] == seeds, "measurement seeds differ")
    _require(calibration["fixed_study_seeds"] == seeds, "calibration seeds differ")
    _require(parity["proposed_paired_seeds"] == seeds, "parity seeds differ")
    _require(measurement["paired_design"]["arms"] == arms, "measurement arms differ")
    _require(causal == EXPECTED_CAUSAL, "source contrasts changed")
    _require(acceptance_causal == causal, "acceptance contrasts differ")
    _require(parity_causal == causal, "causal parity contrasts differ")
    _require(source["future_design"]["tasks"] == ["T1", "T2", "T3", "T4"], "task order differs")
    _require(calibration["proposed_task_ids"] == source["future_design"]["tasks"],
             "calibration task order differs")
    _require(source["future_design"]["checkpoints"] ==
             ["pretrain", "post_T1", "post_T2", "post_T3", "post_T4"],
             "checkpoint order differs")
    _require(source["future_design"]["maximum_updates_per_arm_seed"] == 512,
             "update budget differs")
    _require(source["future_design"]["test_tuning_allowed"] is False,
             "heldout tuning enabled")
    _require(calibration["selection_fences"]["heldout_labels_for_hyperparameters"] is False,
             "heldout labels could influence tuning")
    _require(calibration["calibration_split"]["heldout_test_records_eligible"] is False,
             "heldout records could enter calibration")
    _require(calibration["calibration_split"]["heldout_test_labels_eligible"] is False,
             "heldout labels could enter calibration")
    _require(acceptance["quality_gate"]["final_mean_accuracy_floor_per_seed_and_arm"] ==
             source["preregistered_quality"]["final_mean_accuracy_floor"],
             "quality floor mismatch")
    _require(acceptance["quality_gate"]["maximum_absolute_final_mean_paired_accuracy_gap_per_seed"] ==
             source["preregistered_quality"]["paired_accuracy_difference_max_abs"],
             "quality gap mismatch")

    gates = source["open_gates"]
    _require(len(gates) == 9 and len(set(gates)) == 9, "missing or duplicate gates")
    _require(gates == [item["id"] for item in readiness["minimum_gates"]],
             "readiness gates differ")
    _require(gates == acceptance["gates_still_open"], "acceptance gates differ")
    _require(all(item["status"] == "OPEN" for item in readiness["minimum_gates"]),
             "readiness gate prematurely closed")
    findings = critique["findings"]
    _require(len(findings) == 6 and len({f["id"] for f in findings}) == 6 and
             all(f["status"] == "OPEN" for f in findings),
             "scientific findings not all open")
    _require(source["measurement_scope"]["hardware_energy"] == "NOT_MEASURED" and
             measurement["physical_energy"] == "NOT_MEASURED",
             "energy status changed")
    _require(reproducibility["results_recorded"] == 0 and
             measurement["any_observations_recorded"] is False,
             "unverified experiment results present")

    return {"decision": "NO_GO", "seeds": len(seeds), "arms": len(arms),
            "causal_contrasts": len(causal), "open_gates": len(gates),
            "open_scientific_findings": len(findings), "model_runs": 0,
            "physical_energy": "NOT_MEASURED"}


if __name__ == "__main__":
    print(json.dumps(reconcile(), sort_keys=True))
