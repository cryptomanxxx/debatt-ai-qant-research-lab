"""CL-4E analysis of manually supplied *synthetic* paired timing fixtures.

No clocks, model calls, training, benchmarking, or energy measurement occur.
This checks a proposed future ABBA instrumentation-overhead protocol.
"""
from __future__ import annotations

from statistics import median

from integrations.continual_learning.cl3y_readiness_audit import scientific_readiness_audit

PATTERN = ("plain", "instrumented", "instrumented", "plain")
SCOPE = "proposed_unit_forward_weight_getitem_overhead_not_model_training"


def analyze_synthetic_pairs(rounds: list[dict]) -> dict:
    """Compute descriptive paired overhead from fixed, synthetic nanosecond samples."""
    if not isinstance(rounds, list) or not 2 <= len(rounds) <= 20:
        raise ValueError("need 2–20 synthetic paired ABBA rounds")
    audit = scientific_readiness_audit()
    if audit["decision"] != "NO_GO" or audit["open_count"] != 9:
        raise ValueError("CL-3Y gates changed")
    absolute = []
    ratios = []
    for n, row in enumerate(rounds):
        if not isinstance(row, dict) or set(row) != {"round", "scope", "origin", "samples"}:
            raise ValueError("round metadata mismatch")
        if type(row["round"]) is not int or row["round"] != n:
            raise ValueError("round order mismatch")
        if row["scope"] != SCOPE or row["origin"] != "SYNTHETIC_FIXTURE":
            raise ValueError("cannot classify real timing data as synthetic")
        samples = row["samples"]
        if not isinstance(samples, list) or len(samples) != 4:
            raise ValueError("ABBA requires four samples")
        for index, sample in enumerate(samples):
            if not isinstance(sample, dict) or set(sample) != {"mode", "elapsed_ns"}:
                raise ValueError("sample schema mismatch")
            if sample["mode"] != PATTERN[index] or type(sample["elapsed_ns"]) is not int or sample["elapsed_ns"] <= 0:
                raise ValueError("ABBA ordering or positive integer durations violated")
        plain = (samples[0]["elapsed_ns"] + samples[3]["elapsed_ns"]) / 2
        traced = (samples[1]["elapsed_ns"] + samples[2]["elapsed_ns"]) / 2
        absolute.append(traced - plain)
        ratios.append(traced / plain)
    return {
        "kind": "CL4E_SYNTHETIC_TIMING_ANALYSIS_ONLY",
        "decision": "NO_GO",
        "round_count": len(rounds),
        "median_overhead_ns": median(absolute),
        "median_ratio": median(ratios),
        "scope": SCOPE,
        "observed_model_runtime": False,
        "instrumentation_overhead_measured": False,
        "energy_measured": False,
        "training_allowed": False,
    }
