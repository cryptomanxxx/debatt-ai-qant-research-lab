"""CL-8A unsigned/unsent scientific review request audit; no outreach."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from integrations.continual_learning.cl7f_h1_decision_audit import (
    PATH as CL7F, audit as audit_cl7f,
)
from integrations.continual_learning.cl7a_scientific_readiness_review import (
    EXPECTED_FINDINGS,
)

PATH = Path(__file__).with_name("cl8a_independent_h1_review_request.json")
FROZEN_PACKET_BLOB_SHA = "a2c065a1f0a06fb4d454db6a62d72d16ea61be9e"


def validate_packet_bytes(content: bytes) -> None:
    """Pin the human-facing packet by exact Git blob digest, not keywords."""
    if not isinstance(content, bytes):
        raise ValueError("review packet must be bytes")
    payload = b"blob " + str(len(content)).encode("ascii") + bytes([0]) + content
    if hashlib.sha1(payload).hexdigest() != FROZEN_PACKET_BLOB_SHA:
        raise ValueError("review packet changed without an independently reviewed update")
EXPECTED_FIELDS = {
    "kind", "version", "decision", "packet", "review_status",
    "independent_reviewer_identified", "independent_scientific_signoff",
    "reviewer_contact_authorized", "reviewer_contact_attempted",
    "review_received", "scientific_go_authorized",
    "dataset_generation_authorized", "model_inference_authorized",
    "model_training_authorized", "model_execution_occurred",
    "new_experiment_results", "physical_energy_measured",
    "parent_decision_matrix", "open_parent_gates_expected",
    "open_scientific_findings_expected", "review_questions",
    "requested_review_outcomes", "required_reviewer_disclosures",
    "review_claim_limit", "next_decision",
}
EXPECTED = {
    "kind": "CL8A_INDEPENDENT_H1_REVIEW_REQUEST_NOT_SENT",
    "version": "cl8a-v1", "decision": "NO_GO",
    "packet": "CL8A_INDEPENDENT_H1_REVIEW_PACKET.md",
    "review_status": "REVIEW_NOT_REQUESTED",
    "parent_decision_matrix": CL7F.name,
    "open_parent_gates_expected": 9,
    "open_scientific_findings_expected": 6,
    "review_questions": list(EXPECTED_FINDINGS),
    "requested_review_outcomes": [
        "accept", "revise", "reject", "unable_to_judge",
    ],
    "required_reviewer_disclosures": [
        "reviewer_identity_or_anonymity_rationale", "relevant_expertise",
        "conflicts_of_interest", "dated_scientific_reasoning",
        "explicit_review_scope",
    ],
    "review_claim_limit": (
        "Scientific review is neither Codex approval nor permission to run "
        "models or claim energy savings."
    ),
    "next_decision": (
        "Marcus may choose whether to contact a qualified independent "
        "researcher with this packet; no outreach is implied."
    ),
}
FALSE_FIELDS = (
    "independent_reviewer_identified", "independent_scientific_signoff",
    "reviewer_contact_authorized", "reviewer_contact_attempted",
    "review_received", "scientific_go_authorized",
    "dataset_generation_authorized", "model_inference_authorized",
    "model_training_authorized", "model_execution_occurred",
    "new_experiment_results", "physical_energy_measured",
)


def validate_request(document: dict, parent_report: dict) -> dict:
    if (parent_report.get("decision") != "NO_GO"
            or parent_report.get("open_cl6a_gates") != 9
            or parent_report.get("open_cl7a_findings") != 6):
        raise ValueError("upstream decision matrix has changed")
    if not isinstance(document, dict) or set(document) != EXPECTED_FIELDS:
        raise ValueError("unrecognized or missing review request fields")
    for key, value in EXPECTED.items():
        if document.get(key) != value:
            raise ValueError(f"invalid external reviewer request claim: {key}")
    for key in FALSE_FIELDS:
        if document.get(key) is not False:
            raise ValueError(f"unapproved scientific signoff or outreach: {key}")
    packet = PATH.with_name(document["packet"])
    if not packet.is_file():
        raise ValueError("referenced reviewer packet is missing")
    validate_packet_bytes(packet.read_bytes())
    return {
        "kind": "CL8A_UNSENT_REVIEW_REQUEST_AUDIT",
        "decision": "NO_GO",
        "review_status": "REVIEW_NOT_REQUESTED",
        "reviewer_contact_attempted": False,
        "independent_scientific_signoff": False,
        "open_parent_gates": parent_report["open_cl6a_gates"],
        "open_scientific_findings": parent_report["open_cl7a_findings"],
        "model_training_executed": False,
        "energy_measured": False,
    }


def audit() -> dict:
    return validate_request(
        json.loads(PATH.read_text(encoding="utf-8")),
        audit_cl7f(),
    )


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2, sort_keys=True))
