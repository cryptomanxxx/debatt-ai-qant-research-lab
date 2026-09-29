"""Fail-closed provenance binding for registered AI Researcher proposal snapshots."""
import hashlib
import json
import re
from pathlib import Path

ARCHIVE = Path("research_queue/ai_researcher/proposals")
DRAFT = Path("research_queue/ai_researcher/latest.json")


def verified_proposal(raw=None, archive=ARCHIVE):
    raw = DRAFT.read_bytes() if raw is None else raw
    proposal = json.loads(raw.decode("utf-8"))
    pid = proposal.get("proposal_id")
    if not isinstance(pid, str) or not re.fullmatch(r"research-[0-9]{8}T[0-9]{12}Z-[0-9a-f]{32}", pid):
        raise ValueError("registered proposal_id missing or malformed")
    digest = hashlib.sha256(raw).hexdigest()
    snapshot = Path(archive) / (pid + ".json")
    if snapshot.read_bytes() != raw:
        raise ValueError("immutable archived proposal differs from reviewed latest draft")
    return {"proposal_id": pid, "proposal_sha256": digest}


def verify_binding(record, expected):
    if any(record.get(key) != expected[key] for key in ("proposal_id", "proposal_sha256")):
        raise ValueError("proposal lineage ID/digest mismatch")
