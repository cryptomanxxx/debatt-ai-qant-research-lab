"""CL-4A offline H1 measurement contract: records only, no model execution.

These guards describe future observed CPU data; validating a fabricated fixture
is not a benchmark, training approval, or hardware energy measurement.
"""
from __future__ import annotations

import json
from pathlib import Path

from integrations.continual_learning.cl3y_readiness_audit import scientific_readiness_audit

PLAN_PATH = Path(__file__).with_name("cl4a_h1_measurement_plan.json")
COUNTERS = (
    "forward_weight_terms", "parameter_reads", "parameter_writes",
    "input_visits", "index_reads", "index_writes", "nonlinearity_counts",
)
SYSTEM = ("cpu_wall_seconds_training", "cpu_wall_seconds_inference", "peak_process_rss_bytes")
ARMS = ("local_hybrid", "sparse_local_hybrid")


def readiness_contract() -> dict:
    """Validate the proposed measurement boundary without closing scientific gates."""
    plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    audit = scientific_readiness_audit()
    if plan["status"] != "REVIEW_ONLY_NO_EXECUTION" or plan["decision"] != "NO_GO":
        raise ValueError("CL-4A must remain review-only")
    if plan["training_allowed"] is not False or plan["inference_allowed"] is not False:
        raise ValueError("CL-4A must not execute models")
    if plan["network_allowed"] is not False or plan["external_calls_max"] != 0:
        raise ValueError("CL-4A must remain offline")
    if plan["energy_status"] != "NOT_MEASURED" or plan["joules_field"] is not None:
        raise ValueError("CL-4A cannot claim measured energy")
    if tuple(plan["primary_arms"]) != ARMS or plan["comparison_type"] != "paired_same_local_rule":
        raise ValueError("CL-4A primary contrast drift")
    if tuple(plan["observed_counter_fields"]) != COUNTERS or tuple(plan["system_fields"]) != SYSTEM:
        raise ValueError("CL-4A measurement schema drift")
    if plan["counter_scope"] != "observed_instrumented_source_events_not_hardware_memory_traffic":
        raise ValueError("CL-4A must not equate logical events with hardware traffic")
    if plan["quality_gate"] != "required_before_cost_superiority_claim":
        raise ValueError("CL-4A must retain quality parity")
    if plan["remaining_gates"] != [row["gate_id"] for row in audit["gates"]]:
        raise ValueError("CL-4A must retain all open CL-3Y gates")
    if audit["open_count"] != 9 or audit["decision"] != "NO_GO":
        raise ValueError("CL-3Y no longer matches CL-4A")
    return {
        "kind": "CL4A_H1_MEASUREMENT_READINESS_ONLY",
        "decision": "NO_GO",
        "open_gates": audit["open_count"],
        "training_allowed": False,
        "inference_allowed": False,
        "energy_measured": False,
    }


def validate_observation(record: dict) -> dict:
    """Strictly validate a future row; does not collect or generate any observation."""
    required = {"phase", "arm", "seed", "task", "checkpoint", "counter_scope",
                "counters", "system", "energy_joules", "quality_status", "source_sha256"}
    if not isinstance(record, dict) or set(record) != required:
        raise ValueError("observation fields missing or unexpected")
    if record["phase"] not in ("training", "inference") or record["arm"] not in ARMS:
        raise ValueError("unknown phase or arm")
    if type(record["seed"]) is not int or record["seed"] not in (101, 102, 103, 104, 105):
        raise ValueError("seed not preregistered")
    if record["task"] not in ("T1", "T2", "T3", "T4") or record["checkpoint"] not in (
        "before", "after_T1", "after_T2", "after_T3", "after_T4"
    ):
        raise ValueError("task/checkpoint not preregistered")
    if record["counter_scope"] != "observed_instrumented_source_events_not_hardware_memory_traffic":
        raise ValueError("unverified counter scope")
    counters = record["counters"]
    if not isinstance(counters, dict) or set(counters) != set(COUNTERS):
        raise ValueError("counter schema mismatch")
    if any(type(counters[k]) is not int or counters[k] < 0 for k in COUNTERS):
        raise ValueError("event counts must be nonnegative integers")
    system = record["system"]
    if not isinstance(system, dict) or set(system) != set(SYSTEM):
        raise ValueError("system schema mismatch")
    # Null means not observed; zero means actually measured as zero, never fabricated.
    for field in SYSTEM:
        value = system[field]
        if value is not None and (type(value) not in (int, float) or not 0 <= value < float("inf")):
            raise ValueError("invalid system observation")
    if record["energy_joules"] is not None:
        raise ValueError("energy is not measured or authorized in CL-4A")
    if record["quality_status"] != "NOT_EVALUATED":
        raise ValueError("CL-4A has not established heldout quality")
    if not isinstance(record["source_sha256"], str) or len(record["source_sha256"]) != 64 or any(
        c not in "0123456789abcdef" for c in record["source_sha256"]
    ):
        raise ValueError("source digest must be canonical lowercase SHA-256")
    return {"valid": True, "energy_status": "NOT_MEASURED", "quality_status": "NOT_EVALUATED"}
