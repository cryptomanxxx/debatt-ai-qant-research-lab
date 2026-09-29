"""Offline, non-executable evaluation plan for AI Researcher v3 proposals.

This does NOT run, dispatch, approve or predict any experiment. It separates
the historical Dev-4 reference gate from strict control outperformance and
rejects fabricated evidence by reloading independently pinned read-only data.
"""
import hashlib
import json

from scripts import research_memory_v3 as memory
from scripts.dev4_round2_isolated_feedback import PINNED_BLOBS, git_blob_sha
from scripts.scientific_reasoning_v3 import _encoded, validate_proposal

CORRECT_OUT_OF = 500  # Five ECG200 test evaluations, 100 predictions per seed.
MAX_PLAN_BYTES = 8192


def _pinned_reference(strategy):
    """Revalidate both frozen benchmark files; never mutate them."""
    memory._strategy(strategy)
    # Verifies original artifact digests and independently pinned Git blob SHA1s.
    official = memory.search_experiments(
        strategy=strategy, limit=memory.MAX_LIMIT)["results"]
    if not official:
        raise ValueError("missing independently pinned benchmark evidence")

    protocols = []
    provenance = []
    for round_number, filename in memory.ROUNDS:
        raw = (memory.BENCH / filename).read_bytes()
        if git_blob_sha(raw) != PINNED_BLOBS[filename]:
            raise ValueError("historical protocol source changed")
        data = json.loads(raw)
        if data["source"]["backend"] != "qant-cpu/software-simulation":
            raise ValueError("incorrect historical benchmark backend")
        evaluation, control = data["evaluation"], data["control"]
        values = (
            evaluation["control_local_width"], evaluation["gate_margin_correct"],
            tuple(evaluation["seeds"]), evaluation["epochs"],
            control["candidate_correct"], control["candidate_parameters"],
        )
        if any(type(n) is not int for n in (values[0], values[1], values[3], values[4], values[5])):
            raise ValueError("non-integer benchmark protocol")
        if values[0] != 16 or values[1] != 10 or values[2] != (301, 302, 303, 304, 305):
            raise ValueError("unexpected frozen comparison settings")
        if values[3] != 100 or not 0 <= values[4] <= CORRECT_OUT_OF or values[5] <= 0:
            raise ValueError("invalid pinned comparator")
        protocols.append(values)
        provenance.append({
            "round": round_number,
            "path": "research_queue/benchmarks/" + filename,
            "git_blob_sha1": PINNED_BLOBS[filename],
            "original_artifact_sha256": data["source"]["artifact_sha256"],
        })
    if len(protocols) != 2 or protocols[0] != protocols[1]:
        raise ValueError("historical comparison protocols disagree")
    if any(row["control_correct"] != protocols[0][4] or
           row["control_parameters"] != protocols[0][5] for row in official):
        raise ValueError("record comparator differs from pinned protocol")
    return official, protocols[0], provenance


def build_evaluation_plan(*, proposal, observed, strategy="gpt-oss-120b"):
    """Create a measurable preregistration *draft* from observed verified data.

    The only allowable observed records are exact byte-equivalent structured
    records from Research Memory's own strategy and completed rounds 1/2.
    A proposal never becomes an observed or completed result by being planned.
    """
    official, reference, provenance = _pinned_reference(strategy)
    if type(observed) is not list or not 1 <= len(observed) <= len(official):
        raise ValueError("invalid observed evidence collection")
    canonical = {row["experiment_id"]: row for row in official}
    evidence = {}
    for row in observed:
        if type(row) is not dict:
            raise ValueError("invalid observed record")
        identifier = row.get("experiment_id")
        if type(identifier) is not str or identifier in evidence or identifier not in canonical:
            raise ValueError("unknown or repeated observed experiment ID")
        if _encoded(row) != _encoded(canonical[identifier]):
            raise ValueError("observed evidence differs from pinned historical record")
        evidence[identifier] = row
    validated = validate_proposal(proposal, evidence, strategy)
    if validated["status"] != "proposal_only":
        raise ValueError("no validated proposal available")

    control_width, gate_margin, seeds, epochs, control_correct, control_parameters = reference
    gate_at_least = control_correct - gate_margin
    highest_observed = max(row["candidate_correct"] for row in observed)
    plan = {
        "schema_version": 1,
        "record_kind": "unexecuted_evaluation_plan",
        "status": "planning_only",
        "strategy": strategy,
        "proposal_sha256": hashlib.sha256(_encoded(proposal)).hexdigest(),
        "observed_verified_ids": sorted(evidence),
        "candidate": {
            "dataset": proposal["dataset"],
            "local_width": proposal["local_width"],
            "epochs": proposal["epochs"],
            "comparison_seeds": list(seeds),
            "has_completed_own_strategy_observation": any(
                row["local_width"] == proposal["local_width"] for row in observed),
            "measurement_fields": list(proposal["expected_measurements"]),
            "measured_candidate_correct": None,
            "measured_candidate_parameters": None,
        },
        "historical_reference_only": {
            "source_backend": "qant-cpu/software-simulation",
            "control_local_width": control_width,
            "control_correct": control_correct,
            "control_parameters": control_parameters,
            "correct_out_of": CORRECT_OUT_OF,
            "gate_margin_correct": gate_margin,
            "reference_rounds": provenance,
        },
        "thresholds": {
            "historical_gate_pass_at_least_correct": gate_at_least,
            "ties_control_exact_correct": control_correct,
            "strictly_exceeds_control_at_least_correct": control_correct + 1,
            "best_observed_own_strategy_correct": highest_observed,
            "strictly_exceeds_best_observed_at_least_correct": highest_observed + 1,
        },
        "interpretation_rules": [
            "The historical Dev-4 gate is only a comparison reference, not a v3 success criterion or authorization.",
            "Exactly tying the control is not strictly outperforming it.",
            "Approaching the control is undefined until a numeric margin is separately preregistered.",
            "Parameter-efficiency cannot be inferred from width; measure candidate_parameters and prespecify the objective.",
            "Aggregated correct counts alone do not establish uncertainty or statistical significance; retain seed-level results.",
            "Do not infer any expected accuracy or performance trend from two observed widths.",
        ],
        "requires_separate_human_compute_approval": True,
        "training_runs": 0,
        "repository_mutations": 0,
    }
    if len(_encoded(plan)) > MAX_PLAN_BYTES:
        raise ValueError("evaluation plan exceeds byte budget")
    return plan
