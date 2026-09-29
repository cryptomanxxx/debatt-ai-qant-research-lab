"""Strict offline study-draft and future external-results STRUCTURAL validator.

Not an experiment runner, dispatch path, provenance attestation, Research Memory
writer or compute authorization. Imports only the pinned v3 hypothesis ledger,
stdlib and read-only data validators. No Groq / Q.ANT client.
"""
import hashlib
import itertools
import json
import re
import statistics
from pathlib import Path

from scripts.research_memory_v3 import ROOT
from scripts.scientific_reasoning_v3 import _encoded
from scripts.v3_hypothesis_registry import build_offline_study_draft, load_hypotheses

PROTOCOL_PATH = ROOT / "research_queue" / "v3_protocols" / "first_standalone_draft.json"
MAX_PROTOCOL_BYTES = 12000
MAX_RESULT_BYTES = 12000
NEW_SEEDS = (401, 402, 403, 404, 405)
WIDTHS = (10, 11, 16)
PER_FIT_FIELDS = frozenset(("seed", "local_width", "correct_of_100", "parameter_count"))
SUBMISSION_FIELDS = frozenset((
    "schema_version", "record_kind", "protocol_id", "protocol_sha256",
    "source", "rows",
))
SOURCE_FIELDS = frozenset((
    "backend", "workflow_run_id", "approval_record_sha256", "raw_artifact_sha256",
))
HEX_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def protocol_template():
    """Derive the complete allowed protocol from source-pinned hypotheses."""
    registry = load_hypotheses()
    study = build_offline_study_draft()
    history = study["historical_protocol_reference"]
    if (study["status"] != "unapproved_unexecuted" or study["training_runs"] != 0
            or history["source_backend_only"] != "qant-cpu/software-simulation"):
        raise ValueError("historical draft cannot be used for this protocol")
    ordered = sorted(registry["entries"], key=lambda row: row["proposal"]["local_width"])
    hypothesis_refs = [{
        "hypothesis_id": row["hypothesis_id"],
        "local_width": row["proposal"]["local_width"],
        "proposal_sha256": row["proposal_sha256"],
    } for row in ordered]
    if [r["local_width"] for r in hypothesis_refs] != [10, 11]:
        raise ValueError("unexpected hypothesis widths")
    if (history["epochs"], history["learning_rate"], history["batch_size"],
            history["frequencies"], history["control_local_width"]) != (
            100, 0.001, 32, [1, 2], 16):
        raise ValueError("historical protocol drift")
    return {
        "schema_version": 1,
        "record_kind": "standalone_v3_protocol_draft",
        "protocol_id": "v3-ecg200-width10-11-fresh-control16-draft1",
        "status": "draft_unapproved_unexecuted",
        "hypotheses": hypothesis_refs,
        "dataset": "ECG200",
        "evaluation_data": {
            "split": "existing_ECG200_test_set",
            "predictions_per_seed": 100,
            "new_independent_holdout": False,
            "caution": (
                "Seeds 401-405 are new training seeds, NOT independent held-out test data. "
                "Previous hypothesis selection has already involved ECG200 evaluation; "
                "findings are exploratory."
            ),
        },
        "method": {
            "backend": "not_selected",
            "reference_backend_only": history["source_backend_only"],
            "epochs": history["epochs"],
            "learning_rate": history["learning_rate"],
            "batch_size": history["batch_size"],
            "frequencies": list(history["frequencies"]),
            "exploratory_training_seeds": list(NEW_SEEDS),
            "candidate_widths": [10, 11],
            "control_width": 16,
            "shared_fresh_control_per_seed": True,
            "expected_model_fits_if_separately_approved": 15,
        },
        "analysis": {
            "primary_descriptive_endpoint": (
                "paired_correct_difference_per_seed = candidate_correct_of_100 - "
                "fresh_control_correct_of_100"
            ),
            "aggregate": (
                "For each candidate width, report total paired_correct_difference over "
                "exactly five seeds, candidate correct out of 500, and fresh control "
                "correct out of 500."
            ),
            "descriptive_outcome_labels": [
                "above_fresh_control", "ties_fresh_control", "below_fresh_control",
            ],
            "uncertainty": (
                "Show all five per-seed paired differences, their range and sample "
                "standard deviation. n=5 and reused ECG200 test data do not justify "
                "claims of confirmatory independence or statistical significance."
            ),
            "parameter_tradeoff": (
                "Report actual parameter_count per width and the accuracy/parameter "
                "Pareto relation versus the fresh control. Never infer parameters "
                "from width alone."
            ),
            "multiplicity": (
                "Width 10 and width 11 are two previously selected hypotheses; "
                "report both comparisons, with no unadjusted confirmatory winner claim."
            ),
            "historical_reference_only": (
                "The old Dev-4 gate 442/500, historical control 452/500 and strict "
                "historic exceedance >=453/500 are descriptive context, NOT "
                "prospective v3 success thresholds."
            ),
        },
        "result_contract": {
            "expected_width_seed_rows": 15,
            "per_fit_fields": [
                "seed", "local_width", "correct_of_100", "parameter_count",
            ],
            "complete_cartesian_width_seed_grid_required": True,
            "exact_one_shared_fresh_control_per_seed": True,
            "artifact_authenticity": (
                "Separate post-run provenance verification required; a structurally "
                "validated result is not completed Research Memory evidence."
            ),
        },
        "execution": {
            "status": "not_authorized",
            "approval_record": None,
            "selected_backend": None,
            "runnable_workflow": None,
            "training_runs": 0,
            "groq_calls": 0,
            "modifies_dev4": False,
            "requires_separate_human_compute_approval": True,
        },
    }


def load_protocol(path=PROTOCOL_PATH):
    """Reject any changed draft, unauthorized backend or executable setting."""
    raw = Path(path).read_bytes()
    if len(raw) > MAX_PROTOCOL_BYTES:
        raise ValueError("protocol draft exceeds byte budget")
    data = json.loads(raw)
    if type(data) is not dict or _encoded(data) != _encoded(protocol_template()):
        raise ValueError("protocol draft differs from source-pinned offline template")
    return data


def protocol_sha256(path=PROTOCOL_PATH):
    return hashlib.sha256(_encoded(load_protocol(path))).hexdigest()


def _valid_sha(value):
    return type(value) is str and HEX_SHA256.fullmatch(value) is not None


def inspect_future_result_structure(submission, *, protocol_path=PROTOCOL_PATH):
    """Inspect externally supplied rows; never verify provenance or import evidence.

    A plausible workflow ID and hexadecimal hashes are *untrusted strings*,
    not proof that the data were collected, the artifact exists, or approval
    was granted. A separate future authenticated intake is required.
    """
    protocol = load_protocol(protocol_path)
    if type(submission) is not dict or set(submission) != SUBMISSION_FIELDS:
        raise ValueError("invalid external results submission shape")
    if len(_encoded(submission)) > MAX_RESULT_BYTES:
        raise ValueError("external result submission exceeds byte budget")
    if (type(submission["schema_version"]) is not int or submission["schema_version"] != 1
            or submission["record_kind"] != "v3_unverified_external_seed_results"
            or submission["protocol_id"] != protocol["protocol_id"]
            or submission["protocol_sha256"] != protocol_sha256(protocol_path)):
        raise ValueError("submission not tied to exact unapproved v3 draft")
    source = submission["source"]
    if type(source) is not dict or set(source) != SOURCE_FIELDS:
        raise ValueError("invalid unverified source description")
    if (source["backend"] != "qant-cpu/software-simulation"
            or type(source["workflow_run_id"]) is not int or source["workflow_run_id"] <= 0
            or not _valid_sha(source["approval_record_sha256"])
            or not _valid_sha(source["raw_artifact_sha256"])):
        raise ValueError("unsupported backend or malformed unverified source claims")

    rows = submission["rows"]
    if type(rows) is not list or len(rows) != 15:
        raise ValueError("expected exactly 15 per-fit seed/width records")
    expected = set(itertools.product(NEW_SEEDS, WIDTHS))
    observed = {}
    params_by_width = {}
    for row in rows:
        if type(row) is not dict or set(row) != PER_FIT_FIELDS:
            raise ValueError("invalid per-fit measurement schema")
        seed, width = row["seed"], row["local_width"]
        correct, params = row["correct_of_100"], row["parameter_count"]
        if (type(seed) is not int or type(width) is not int
                or (seed, width) not in expected or (seed, width) in observed):
            raise ValueError("duplicate, unknown or historical seed/width combination")
        if (type(correct) is not int or not 0 <= correct <= 100
                or type(params) is not int or params <= 0):
            raise ValueError("invalid external measurement")
        if width in params_by_width and params_by_width[width] != params:
            raise ValueError("parameter count varied for the same fixed architecture width")
        params_by_width[width] = params
        observed[seed, width] = correct
    if set(observed) != expected:
        raise ValueError("missing paired candidate/control measurement")

    fresh_control_total = sum(observed[seed, 16] for seed in NEW_SEEDS)
    comparisons = []
    for width in (10, 11):
        differences = [
            observed[seed, width] - observed[seed, 16] for seed in NEW_SEEDS
        ]
        delta = sum(differences)
        candidate_correct = sum(observed[seed, width] for seed in NEW_SEEDS)
        if candidate_correct != fresh_control_total + delta:
            raise ValueError("inconsistent paired aggregation")
        control_dominates = (
            fresh_control_total >= candidate_correct
            and params_by_width[16] <= params_by_width[width]
            and (fresh_control_total > candidate_correct
                 or params_by_width[16] < params_by_width[width])
        )
        candidate_dominates = (
            candidate_correct >= fresh_control_total
            and params_by_width[width] <= params_by_width[16]
            and (candidate_correct > fresh_control_total
                 or params_by_width[width] < params_by_width[16])
        )
        comparisons.append({
            "candidate_width": width,
            "candidate_correct_out_of_500_unverified": candidate_correct,
            "fresh_control_correct_out_of_500_unverified": fresh_control_total,
            "paired_correct_differences_by_seed_unverified": [
                {"seed": seed, "correct_difference": d}
                for seed, d in zip(NEW_SEEDS, differences)
            ],
            "total_paired_correct_difference_unverified": delta,
            "paired_difference_range_unverified": [min(differences), max(differences)],
            "sample_sd_paired_differences_unverified": statistics.stdev(differences),
            "candidate_parameters_unverified": params_by_width[width],
            "fresh_control_parameters_unverified": params_by_width[16],
            "descriptive_accuracy_relation_unverified": (
                "above_fresh_control" if delta > 0 else
                "ties_fresh_control" if delta == 0 else "below_fresh_control"
            ),
            "pareto_relation_to_fresh_control_unverified": (
                "candidate_dominates_control" if candidate_dominates
                else "control_dominates_candidate" if control_dominates
                else "tradeoff_or_exact_tie"
            ),
        })
    return {
        "schema_version": 1,
        "status": "structural_checks_only_unverified",
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_sha256(protocol_path),
        "claimed_backend": source["backend"],
        "comparisons": comparisons,
        "authenticity_verified": False,
        "compute_approval_verified": False,
        "independent_holdout_data": False,
        "eligible_for_research_memory": False,
        "training_runs_started_by_validator": 0,
        "repository_writes": 0,
        "requires_separate_human_provenance_review": True,
    }
