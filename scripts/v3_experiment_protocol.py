"""Strict offline study-draft and future external-results STRUCTURAL validator.

Not an experiment runner, dispatch path, provenance attestation, Research Memory
writer or compute authorization. Imports only the pinned v3 hypothesis ledger,
stdlib and read-only data validators. No Groq / Q.ANT client.
"""
import hashlib
import itertools
import json
import math
import re
import statistics
from pathlib import Path

from scripts.research_memory_v3 import ROOT
from scripts.dev4_round2_isolated_feedback import git_blob_sha
from scripts.scientific_reasoning_v3 import _encoded
from scripts.v3_hypothesis_registry import build_offline_study_draft, load_hypotheses

PROTOCOL_PATH = ROOT / "research_queue" / "v3_protocols" / "first_standalone_draft.json"
PRIOR_WIDTH10_REL_PATH = "results/result_engine_w10_100epoch_ai-local-width-w10-100epoch-201-202-203-204-205.json"
PRIOR_WIDTH10_BLOB_SHA1 = "a23d3343777599d9296e708fec52513226fff940"
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



def prior_width10_reference():
    """A repository-recorded completed WIDTH 10 study, outside isolated Dev-4 memory.

    Prior experiment seeds 201-205 predate the new v3 hypothesis. Pin the Git
    blob, validate its configuration, and independently reconcile per-seed
    accuracies with published summary totals. A repository record is not
    external provenance attestation or photonic hardware evidence.
    """
    raw = (ROOT / PRIOR_WIDTH10_REL_PATH).read_bytes()
    if len(raw) > 32768 or git_blob_sha(raw) != PRIOR_WIDTH10_BLOB_SHA1:
        raise ValueError("previous width10 source changed or too large")
    source = json.loads(raw)
    if (type(source) is not dict or source.get("schema_version") != 1
            or source.get("backend") != "qant-cpu/software-simulation"
            or source.get("dataset", {}).get("name") != "ECG200"
            or source.get("dataset", {}).get("test_shape") != [100, 96]):
        raise ValueError("invalid prior width10 source")
    config = source["configuration"]
    if (config["seeds"] != [201, 202, 203, 204, 205]
            or type(config["epochs"]) is not int or config["epochs"] != 100
            or type(config["learning_rate"]) is not float
            or config["learning_rate"] != 0.001
            or type(config["batch_size"]) is not int or config["batch_size"] != 32
            or config["frequencies"] != [1, 2]
            or config["candidates"] != {
                "alpha5_w10": [10, 3], "alpha5_w16_control": [16, 3]}):
        raise ValueError("prior width10 study has incompatible settings")
    rows = source["rows"]
    if type(rows) is not list or len(rows) != 10:
        raise ValueError("invalid prior width10 seed rows")
    totals, parameters = {}, {}
    for candidate in ("alpha5_w10", "alpha5_w16_control"):
        records = [row for row in rows if row.get("candidate") == candidate]
        if len(records) != 5 or {r["seed"] for r in records} != set(range(201, 206)):
            raise ValueError("prior width10 has incomplete or repeated seed pairs")
        scores, param_values = [], set()
        for row in records:
            accuracy, params = row["qant_accuracy"], row["parameter_count"]
            if (type(accuracy) is not float or not 0 <= accuracy <= 1
                    or type(params) is not int or params <= 0
                    or not math.isclose(100 * accuracy, round(100 * accuracy),
                                        abs_tol=1e-8)):
                raise ValueError("invalid prior width10 recorded score")
            scores.append(round(100 * accuracy))
            param_values.add(params)
        if len(param_values) != 1:
            raise ValueError("prior width10 inconsistent parameter counts")
        totals[candidate] = sum(scores)
        parameters[candidate] = param_values.pop()
        summary = source["summary"][candidate]
        if (type(summary["aggregate_qant_correct"]) is not int
                or summary["aggregate_qant_correct"] != totals[candidate]
                or type(summary["parameter_count"]) is not int
                or summary["parameter_count"] != parameters[candidate]):
            raise ValueError("prior width10 source row-summary mismatch")
    if (totals != {"alpha5_w10": 444, "alpha5_w16_control": 453}
            or parameters != {"alpha5_w10": 5346, "alpha5_w16_control": 8550}):
        raise ValueError("unexpected prior width10 benchmark comparison")
    return {
        "classification": "previous_recorded_width10_test_outside_isolated_own_strategy_memory",
        "source_path": PRIOR_WIDTH10_REL_PATH,
        "source_git_blob_sha1": PRIOR_WIDTH10_BLOB_SHA1,
        "experiment_id": source["experiment_id"],
        "proposal_id": source["proposal_id"],
        "backend": source["backend"],
        "seeds": list(config["seeds"]),
        "epochs": config["epochs"],
        "learning_rate": config["learning_rate"],
        "batch_size": config["batch_size"],
        "frequencies": list(config["frequencies"]),
        "candidate_width": 10,
        "candidate_correct_out_of_500": totals["alpha5_w10"],
        "candidate_parameter_count": parameters["alpha5_w10"],
        "paired_control_width": 16,
        "paired_control_correct_out_of_500": totals["alpha5_w16_control"],
        "paired_control_parameter_count": parameters["alpha5_w16_control"],
        "paired_correct_difference": (
            totals["alpha5_w10"] - totals["alpha5_w16_control"]),
        "provenance_scope": (
            "Completed repository JSON record, not in frozen isolated Dev-4 "
            "strategy memory; no independent external artifact attestation."),
    }


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
        "prior_width10_repository_result": prior_width10_reference(),
        "dataset": "ECG200",
        "evaluation_data": {
            "split": "existing_ECG200_test_set",
            "predictions_per_seed": 100,
            "new_independent_holdout": False,
            "caution": (
                "Seeds 401-405 are new training seeds, NOT independent held-out test data. "
                "A prior completed width-10 ECG200 repository result used training seeds 201-205, "
                "and other previous hypothesis selection involved ECG200 evaluation; "
                "findings are exploratory, not a new independent holdout."
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
            "candidate_roles": {
                "10": "replication_of_prior_repository_width10_configuration_new_training_seeds",
                "11": "new_v3_hypothesis_not_completed_in_isolated_own_strategy_memory",
            },
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
                "Width 10 repeats a previously tested architecture using new paired "
                "training seeds; width 11 is the other v3 hypothesis. Report both "
                "exploratory comparisons without an unadjusted confirmatory winner claim."
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
