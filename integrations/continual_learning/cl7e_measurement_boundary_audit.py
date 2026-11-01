"""CL-7E strictly static measurement-boundary review. No model/hardware access."""
from __future__ import annotations

import json
from pathlib import Path

from integrations.continual_learning.cl6a_readiness_review import PATH as CL6A, review
from integrations.continual_learning.cl7a_scientific_readiness_review import (
    PATH as CL7A, validate_critique,
)
from integrations.continual_learning.cl7b_h1_baseline_audit import (
    PATH as CL7B, validate_contract as validate_baselines,
)
from integrations.continual_learning.cl7c_train_only_calibration_audit import (
    PATH as CL7C, validate_protocol as validate_calibration,
)
from integrations.continual_learning.cl7d_causal_pairing_audit import (
    PATH as CL7D, validate_contract as validate_pairing,
)

PATH = Path(__file__).with_name("cl7e_measurement_boundaries.json")
EXPECTED_COUNTERS = [
    ("source_logical_events", "PROXY_ONLY",
     ["logical_reads", "logical_writes", "index_visits", "nonlinearity_calls", "control_visits"],
     "Static algorithmic event counts are not DRAM transactions, elapsed time, or joules."),
    ("cpu_elapsed_time", "NOT_MEASURED", ["monotonic_elapsed_seconds"],
     "A future instrumented CPU timing run would require fixed hardware/runtime, repetition, warmup and confidence reporting."),
    ("process_memory", "NOT_MEASURED", ["peak_rss_bytes"],
     "Process RSS is a resource proxy, not physical memory bandwidth or per-weight DRAM traffic."),
    ("electrical_energy", "NOT_MEASURED", ["joules"],
     "Requires separately approved meter, measured idle baseline, sampling window and host boundary."),
    ("photonic_or_neuromorphic_hardware", "NOT_MEASURED", [],
     "No Q.ANT, Loihi or Hala Point performance or joule evidence has been obtained."),
]
EXPECTED_RULES = {
    "training_and_inference_separate": True,
    "per_seed_and_per_arm_scope": True,
    "equal_task_order_and_workload_required": True,
    "instrumented_vs_uninstrumented_control_required": True,
    **{field: "UNSPECIFIED_BLOCKING" for field in (
        "wall_clock_repeat_count", "warmup_schedule",
        "timing_summary_and_uncertainty", "cpu_and_software_environment",
        "observer_overhead_bound", "power_meter_and_idle_baseline")},
    "missing_samples_policy": "REPORT_ALL_AND_SUSPEND_CLAIMS",
}
EXPECTED_FENCES = {
    "logical_bytes_equated_to_dram_bytes": False,
    "logical_operations_equated_to_joules": False,
    "runtime_equated_to_electrical_energy": False,
    "cpu_result_extrapolated_to_photonic": False,
    "quality_matched_savings_claim_permitted": False,
    "physical_energy_claim_permitted": False,
    "scientific_go_authorized": False,
}
EXPECTED_FIELDS = {
    "kind", "version", "decision", "analysis_only", "parent_protocol",
    "critique", "baseline_contract", "calibration_protocol", "causal_protocol",
    "model_construction_authorized", "model_inference_authorized",
    "training_authorized", "dataset_generation_authorized", "new_results_present",
    "independent_scientific_signoff", "hardware_energy_measured", "counter_classes",
    "future_observation_rules", "interpretation_fences",
    "open_cl6a_gates_expected", "open_cl7a_findings_expected", "next_decision",
}


def validate_contract(document: dict, parent: dict, critique: dict,
                      baselines: dict, calibration: dict, pairing: dict) -> dict:
    pr = review(parent)
    cr = validate_critique(critique, parent)
    br = validate_baselines(baselines, parent, critique)
    cal = validate_calibration(calibration, parent, critique, baselines)
    pair = validate_pairing(pairing, parent, critique, baselines, calibration)
    if (pr["decision"] != "NO_GO" or pr["closed_gates"] != 0
            or cr["open_scientific_findings"] != 6
            or br["decision"] != "NO_GO" or cal["decision"] != "NO_GO"
            or pair["decision"] != "NO_GO"):
        raise ValueError("prerequisite readiness guards changed")
    if not isinstance(document, dict) or set(document) != EXPECTED_FIELDS:
        raise ValueError("unknown or missing measurement-boundary contract field")
    fixed = {
        "kind": "CL7E_H1_MEASUREMENT_BOUNDARIES_NOT_EXECUTED",
        "version": "cl7e-v1", "decision": "NO_GO", "analysis_only": True,
        "parent_protocol": CL6A.name, "critique": CL7A.name,
        "baseline_contract": CL7B.name, "calibration_protocol": CL7C.name,
        "causal_protocol": CL7D.name, "open_cl6a_gates_expected": 9,
        "open_cl7a_findings_expected": 6,
        "future_observation_rules": EXPECTED_RULES,
        "interpretation_fences": EXPECTED_FENCES,
        "next_decision": "Define and externally review CPU timing/observer protocol; physical energy requires a separate instrumented measurement authorization.",
    }
    for key, value in fixed.items():
        if document.get(key) != value:
            raise ValueError(f"unapproved measurement or execution claim: {key}")
    for key in ("model_construction_authorized", "model_inference_authorized",
                "training_authorized", "dataset_generation_authorized",
                "new_results_present", "independent_scientific_signoff",
                "hardware_energy_measured"):
        if document.get(key) is not False:
            raise ValueError(f"unapproved execution/result flag: {key}")
    counters = document.get("counter_classes")
    if not isinstance(counters, list) or len(counters) != len(EXPECTED_COUNTERS):
        raise ValueError("expected five separate measurement domains")
    for actual, (identifier, status, units, claim) in zip(counters, EXPECTED_COUNTERS):
        if actual != {"id": identifier, "status": status, "units": units, "claim": claim}:
            raise ValueError(f"measurement domain altered: {identifier}")
    return {
        "kind": "CL7E_STATIC_MEASUREMENT_SCOPE_AUDIT",
        "decision": "NO_GO", "measurement_domains": len(counters),
        "physical_energy_measured": False, "cpu_runtime_measured": False,
        "new_experimental_results": False, "training_executed": False,
        "open_parent_gates": len(pr["open_gates"]),
        "open_scientific_findings": cr["open_scientific_findings"],
    }


def audit() -> dict:
    return validate_contract(
        json.loads(PATH.read_text(encoding="utf-8")),
        json.loads(CL6A.read_text(encoding="utf-8")),
        json.loads(CL7A.read_text(encoding="utf-8")),
        json.loads(CL7B.read_text(encoding="utf-8")),
        json.loads(CL7C.read_text(encoding="utf-8")),
        json.loads(CL7D.read_text(encoding="utf-8")),
    )


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2, sort_keys=True))
