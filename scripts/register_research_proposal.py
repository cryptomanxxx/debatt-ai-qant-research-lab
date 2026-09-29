"""Archive immutable, uniquely identified AI Researcher drafts before updating latest.json.

This module records provenance only. It never approves proposals or starts compute.
"""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

ARCHIVE = Path("research_queue/ai_researcher/proposals")


def register(draft, latest, archive=ARCHIVE, *, now=None, token=None):
    """Return a registered snapshot. Refuse to overwrite an existing archive."""
    if not isinstance(draft, dict) or draft.get("status") != "draft_requires_human_review":
        raise ValueError("only reviewable draft objects may be registered")
    if draft.get("proposal_id") is not None:
        raise ValueError("draft already has a proposal_id; refusing to re-register")
    if not isinstance(draft.get("proposal"), dict) or draft["proposal"].get("requires_human_approval") is not True:
        raise ValueError("human approval must remain mandatory")
    stamp = now or datetime.now(timezone.utc)
    if stamp.tzinfo is None:
        raise ValueError("registration timestamp must be timezone-aware")
    stamp = stamp.astimezone(timezone.utc)
    uid = token or uuid4().hex
    if not isinstance(uid, str) or len(uid) != 32 or any(c not in "0123456789abcdef" for c in uid):
        raise ValueError("invalid registration token")
    pid = "research-" + stamp.strftime("%Y%m%dT%H%M%S%fZ") + "-" + uid
    snapshot = dict(draft)
    snapshot["proposal_id"] = pid
    snapshot["created_at_utc"] = stamp.isoformat().replace("+00:00", "Z")
    snapshot["strategy"] = "gpt-oss-120b"
    serialized = (json.dumps(snapshot, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    archive = Path(archive)
    archive.mkdir(parents=True, exist_ok=True)
    path = archive / (pid + ".json")
    # Exclusive creation: a historical proposal is never silently replaced.
    with path.open("xb") as handle:
        handle.write(serialized)
    latest = Path(latest)
    latest.parent.mkdir(parents=True, exist_ok=True)
    # latest is a mutable pointer; the archived snapshot is the evidence record.
    latest.write_bytes(serialized)
    return {"proposal_id": pid, "proposal_file": str(path), "strategy": snapshot["strategy"],
            "sha256": hashlib.sha256(serialized).hexdigest()}
