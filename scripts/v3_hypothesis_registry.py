"""Read-only archive of real AI Researcher v3 hypotheses and a v3 study draft.

NO imports of Groq adapters, job materializers, training or GitHub dispatch.
Historical Dev-4 experiment evidence stays in the existing independently pinned
Research Memory. AI hypotheses are NEVER treated as completed measurements.
"""
import hashlib
import json
from pathlib import Path

from scripts import research_memory_v3 as memory
from scripts.dev4_round2_isolated_feedback import PINNED_BLOBS, git_blob_sha
from scripts.scientific_evaluation_v3 import build_evaluation_plan
from scripts.scientific_reasoning_v3 import _encoded, validate_proposal

REGISTRY = memory.ROOT / "research_queue" / "v3_hypotheses" / "registry.json"
MAX_REGISTRY_BYTES = 16000
SOURCE_PINS = {
    "v3-live-36597959269-width11": {
        "run": 36597959269, "commit": "15442ba85ea33b12528f00e8870f3db2e7c5b7b5",
        "artifact": 11047390153, "member": "research_session.json",
        "report_sha256": "8d436214e35b45fb713b912b64eeee4bd1d10f36667ec0faa7019e5771f9f6a1",
        "proposal_sha256": "d3a53ed7f3b2adcb4e903dcb1803b6a58ec263933e172bc7994808a5f3b13eae",
        "proposal_output_sha256": "985b435b4d28698425e0733405d8791947cfa47b481b343f92eef227ef7a0932",
        "width": 11, "reviewed": False,
    },
    "v3-reviewed-36601207596-width10": {
        "run": 36601207596, "commit": "ec32cf42fe345a6fa0bc0416753dc7b2da9aee57",
        "artifact": 11048933655, "member": "reviewed_research_session.json",
        "report_sha256": "b4c40fa679c573e61eef5edba38301071ef638d3c6ad5380b35749b5d4064fe9",
        "proposal_sha256": "036520e5857c05a76f8326cb1ee11048fa38a266247159e66c469478a9e4e44c",
        "proposal_output_sha256": "1897c8350523e2b90e525e777c8c4250f3cefaa824171c3f252b2aa76ef988b6",
        "width": 10, "reviewed": True,
    },
}
ENTRY_FIELDS = frozenset((
    "hypothesis_id", "state", "strategy", "source", "proposal_sha256",
    "observed_evidence_ids", "proposal", "review", "completed_experiment_id",
    "measured_candidate_correct", "measured_candidate_parameters",
))
SOURCE_FIELDS = frozenset((
    "workflow_run_id", "workflow_commit_sha", "artifact_id", "artifact_member",
    "original_report_sha256", "proposal_model_output_sha256",
))
REVIEW_FIELDS = frozenset(("kind", "assessment", "reviewed_proposal_sha256"))
REGISTRY_FIELDS = frozenset((
    "schema_version", "record_kind", "status", "entries", "training_runs",
    "requires_separate_human_compute_approval",
))
OBSERVED_IDS = ("dev4-r1-gpt-oss-120b", "dev4-r2-gpt-oss-120b")


def _digest(value):
    return hashlib.sha256(value).hexdigest()


def load_hypotheses(path=REGISTRY):
    """Fail closed against edited, fabricated or promoted historical proposals."""
    raw = Path(path).read_bytes()
    if len(raw) > MAX_REGISTRY_BYTES:
        raise ValueError("hypothesis registry exceeds byte budget")
    doc = json.loads(raw)
    if type(doc) is not dict or set(doc) != REGISTRY_FIELDS:
        raise ValueError("invalid hypothesis registry shape")
    if (type(doc["schema_version"]) is not int or doc["schema_version"] != 1
            or doc["record_kind"] != "unexecuted_hypothesis_registry"
            or doc["status"] != "hypotheses_only"
            or doc["training_runs"] != 0
            or doc["requires_separate_human_compute_approval"] is not True):
        raise ValueError("registry cannot authorize or claim completed research")
    rows = doc["entries"]
    if type(rows) is not list or len(rows) != len(SOURCE_PINS):
        raise ValueError("hypothesis registry missing pinned source")
    canonical = {row["experiment_id"]: row for row in
                 memory.search_experiments(strategy="gpt-oss-120b")["results"]}
    if set(canonical) != set(OBSERVED_IDS):
        raise ValueError("unexpected historical evidence inventory")

    seen = set()
    for row in rows:
        if type(row) is not dict or set(row) != ENTRY_FIELDS:
            raise ValueError("invalid hypothesis entry shape")
        key = row["hypothesis_id"]
        if type(key) is not str or key not in SOURCE_PINS or key in seen:
            raise ValueError("unknown or duplicate hypothesis")
        seen.add(key)
        pin, source = SOURCE_PINS[key], row["source"]
        if type(source) is not dict or set(source) != SOURCE_FIELDS:
            raise ValueError("invalid hypothesis source")
        if (type(source["workflow_run_id"]) is not int
                or source["workflow_run_id"] != pin["run"]
                or source["workflow_commit_sha"] != pin["commit"]
                or type(source["artifact_id"]) is not int
                or source["artifact_id"] != pin["artifact"]
                or source["artifact_member"] != pin["member"]
                or source["original_report_sha256"] != pin["report_sha256"]
                or source["proposal_model_output_sha256"] != pin["proposal_output_sha256"]):
            raise ValueError("unrecognized original run or report fingerprint")
        if (row["state"] != "proposed_untested"
                or row["strategy"] != "gpt-oss-120b"
                or row["completed_experiment_id"] is not None
                or row["measured_candidate_correct"] is not None
                or row["measured_candidate_parameters"] is not None):
            raise ValueError("hypothesis cannot masquerade as a measurement")
        if (type(row["proposal_sha256"]) is not str
                or row["proposal_sha256"] != pin["proposal_sha256"]
                or _digest(_encoded(row["proposal"])) != pin["proposal_sha256"]
                or row["proposal"]["local_width"] != pin["width"]):
            raise ValueError("registered proposal differs from original hash")
        if (type(row["observed_evidence_ids"]) is not list
                or row["observed_evidence_ids"] != list(OBSERVED_IDS)):
            raise ValueError("unexpected observed evidence IDs")
        validate_proposal(row["proposal"], canonical, "gpt-oss-120b")
        review = row["review"]
        if type(review) is not dict or set(review) != REVIEW_FIELDS:
            raise ValueError("invalid review metadata")
        expected = ("model_critical_review", "testable_with_caveats",
                    pin["proposal_sha256"]) if pin["reviewed"] else (
                        "not_reviewed", None, None)
        if (review["kind"], review["assessment"],
                review["reviewed_proposal_sha256"]) != expected:
            raise ValueError("review status differs from pinned source")
    if seen != set(SOURCE_PINS):
        raise ValueError("missing pinned hypotheses")
    return doc


def verify_original_report_bytes(raw, hypothesis_entry):
    """Optionally re-check a locally downloaded original Actions report.

    The exact raw file hash is pinned. JSON contents are *also* checked for
    proposal, model-step provenance, strategy, evidence and no training.
    Never downloads anything automatically.
    """
    if type(raw) is not bytes or len(raw) > 32768:
        raise ValueError("invalid original research report bytes")
    key = hypothesis_entry["hypothesis_id"]
    if key not in SOURCE_PINS:
        raise ValueError("unknown original report")
    pin = SOURCE_PINS[key]
    if _digest(raw) != pin["report_sha256"]:
        raise ValueError("original report fingerprint mismatch")
    report = json.loads(raw)
    result = report["result"]
    evidence = report.get("observed_evidence", result.get("evidence"))
    if (result["status"] != "proposal_only" or result["strategy"] != "gpt-oss-120b"
            or report["training_runs"] != 0
            or result["requires_separate_human_compute_approval"] is not True
            or result["proposal"] != hypothesis_entry["proposal"]
            or sorted(x["experiment_id"] for x in evidence) != sorted(OBSERVED_IDS)):
        raise ValueError("original source does not match archived hypothesis")
    proposal_steps = [x for x in report["audit"] if x["kind"] == "proposal"]
    if len(proposal_steps) != 1 or (
            proposal_steps[0]["model_output_sha256"] != pin["proposal_output_sha256"]):
        raise ValueError("proposal model step fingerprint mismatch")
    if pin["reviewed"]:
        review = result["critical_review"]
        if (review["assessment"] != hypothesis_entry["review"]["assessment"]
                or review["reviewed_proposal_sha256"] != pin["proposal_sha256"]
                or report["review_model_calls"] != 1):
            raise ValueError("registered review differs from source")
    return True


def _pinned_protocol():
    """Verify both completed Dev-4 protocol files, without modifying them."""
    settings = []
    for number, filename in memory.ROUNDS:
        raw = (memory.BENCH / filename).read_bytes()
        if git_blob_sha(raw) != PINNED_BLOBS[filename]:
            raise ValueError("protocol source hash mismatch")
        data = json.loads(raw)
        if data["schema_version"] != 1 or data["source"]["backend"] != "qant-cpu/software-simulation":
            raise ValueError("invalid historical protocol provenance")
        ev = data["evaluation"]
        entry = (ev["epochs"], tuple(ev["seeds"]), ev["learning_rate"],
                 ev["batch_size"], tuple(ev["frequencies"]),
                 ev["control_local_width"])
        if (type(entry[0]) is not int or entry[0] != 100
                or entry[1] != (301, 302, 303, 304, 305)
                or type(entry[2]) is not float or entry[2] != 0.001
                or type(entry[3]) is not int or entry[3] != 32
                or entry[4] != (1, 2)
                or type(entry[5]) is not int or entry[5] != 16):
            raise ValueError("unexpected historical hyperparameters")
        settings.append(entry)
    if len(settings) != 2 or settings[0] != settings[1]:
        raise ValueError("historic rounds disagree on protocol")
    return settings[0]


def build_offline_study_draft():
    """Produce two reproducible *planning-only* evaluations and study metadata."""
    registry = load_hypotheses()
    epochs, seeds, learning_rate, batch_size, frequencies, control_width = _pinned_protocol()
    known = {row["experiment_id"]: row for row in
             memory.search_experiments(strategy="gpt-oss-120b")["results"]}
    proposals = []
    for item in registry["entries"]:
        observed = [known[k] for k in item["observed_evidence_ids"]]
        plan = build_evaluation_plan(
            proposal=item["proposal"], observed=observed, strategy=item["strategy"])
        if (plan["status"] != "planning_only" or plan["training_runs"] != 0
                or plan["candidate"]["measured_candidate_correct"] is not None
                or plan["candidate"]["measured_candidate_parameters"] is not None):
            raise ValueError("evaluation unexpectedly contains executed data")
        proposals.append({
            "hypothesis_id": item["hypothesis_id"],
            "state": "proposed_untested",
            "origin_workflow_run_id": item["source"]["workflow_run_id"],
            "source_report_sha256": item["source"]["original_report_sha256"],
            "review_metadata": item["review"],
            "evaluation_plan": plan,
        })
    return {
        "schema_version": 1,
        "record_kind": "offline_study_draft",
        "status": "unapproved_unexecuted",
        "dataset": "ECG200",
        "own_strategy_hypothesis_widths": [
            item["proposal"]["local_width"] for item in registry["entries"]],
        "historical_protocol_reference": {
            "source_backend_only": "qant-cpu/software-simulation",
            "epochs": epochs, "seeds": list(seeds),
            "learning_rate": learning_rate, "batch_size": batch_size,
            "frequencies": list(frequencies), "control_local_width": control_width,
            "control_comparison_correct": 452,
            "control_comparison_parameters": 8550,
        },
        "proposed_future_design_not_authorized": {
            "candidate_widths": sorted(item["proposal"]["local_width"]
                                       for item in registry["entries"]),
            "paired_control_width": control_width,
            "fresh_paired_control_proposed": True,
            "seed_level_fields": [
                "seed", "candidate_width", "candidate_correct",
                "fresh_control_correct", "candidate_parameters",
                "fresh_control_parameters", "paired_correct_difference",
            ],
            "measurements": "Record paired seed-level data for each candidate "
                            "and a newly run control *if and only if separately "
                            "authorized*. Historical control is descriptive only.",
            "primary_endpoint_status": "requires_explicit_preregistration",
            "uncertainty_method_status": "requires_explicit_preregistration",
            "backend_status": "not_selected",
            "success_criteria_status": "requires_explicit_preregistration",
            "automatic_compute_or_dispatch": False,
        },
        "interpretation": [
            "Separate old Dev-4 reference gate (442/500), old control tie "
            "(452/500), and exceeding the old control (453/500).",
            "A future freshly paired control must be evaluated on its own "
            "observed data; never assume it scores 452 again.",
            "Two untested hypotheses are not two additional measurements.",
            "Parameter efficiency needs new measurements and a preregistered "
            "multiobjective or trade-off definition.",
            "Changing to optical hardware would require its own validation "
            "and cannot be mislabeled as a historical software-simulation result.",
        ],
        "proposals": proposals,
        "groq_calls": 0,
        "training_runs": 0,
        "requires_separate_human_compute_approval": True,
        "modifies_frozen_dev4": False,
    }


if __name__ == "__main__":
    print(json.dumps(build_offline_study_draft(), sort_keys=True, indent=2,
                     ensure_ascii=False))
