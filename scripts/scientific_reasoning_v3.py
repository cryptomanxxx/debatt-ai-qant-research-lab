"""Offline Scientific Reasoning Engine; no live model adapter or compute access.

A scripted model supplies structured tool requests and a final proposal. The
trusted application executes only ResearchSession's read-only allowlist and
validates citations against evidence actually returned in this session.
"""
import hashlib
import json
from scripts.research_tool_calling_v3 import ResearchSession
from scripts.research_memory_v3 import _strategy

MAX_STEPS = 6
MAX_TRANSCRIPT_BYTES = 32000
MAX_PROPOSAL_BYTES = 4096
PROPOSAL_FIELDS = frozenset(("hypothesis", "rationale", "local_width", "dataset",
                              "epochs", "evidence_ids", "expected_measurements"))
MEASUREMENTS = frozenset(("candidate_correct", "candidate_parameters", "gate_pass"))


def _encoded(value):
    return json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":")).encode("utf-8")


def validate_proposal(proposal, evidence, strategy):
    """Evidence IDs must have been returned by this session, not merely exist on disk."""
    _strategy(strategy)
    if type(proposal) is not dict or set(proposal) != PROPOSAL_FIELDS:
        raise ValueError("invalid scientific proposal schema")
    if len(_encoded(proposal)) > MAX_PROPOSAL_BYTES:
        raise ValueError("proposal too large")
    for field in ("hypothesis", "rationale"):
        if type(proposal[field]) is not str or not 10 <= len(proposal[field]) <= 1000:
            raise ValueError("invalid scientific text")
    if type(proposal["local_width"]) is not int or not 4 <= proposal["local_width"] <= 15:
        raise ValueError("invalid proposed width")
    if proposal["dataset"] != "ECG200" or type(proposal["epochs"]) is not int or proposal["epochs"] != 100:
        raise ValueError("unsupported experiment configuration")
    ids = proposal["evidence_ids"]
    if type(ids) is not list or not 1 <= len(ids) <= 5 or any(type(x) is not str for x in ids):
        raise ValueError("invalid evidence references")
    if len(set(ids)) != len(ids) or not set(ids).issubset(evidence):
        raise ValueError("proposal cites unobserved evidence")
    if any(evidence[x]["strategy"] != strategy for x in ids):
        raise ValueError("cross-strategy evidence")
    measurements = proposal["expected_measurements"]
    if type(measurements) is not list or set(measurements) != MEASUREMENTS or len(measurements) != 3:
        raise ValueError("required evaluation measurements missing")
    return {"schema_version": 1, "status": "proposal_only", "strategy": strategy,
            "proposal": proposal, "evidence": [evidence[x] for x in ids],
            "requires_separate_human_compute_approval": True}


def run_scripted_session(*, strategy, steps):
    """Simulated model dialogue; rejects extra actions after the final proposal."""
    strategy = _strategy(strategy)
    if type(steps) is not list or not 1 <= len(steps) <= MAX_STEPS:
        raise ValueError("invalid scripted dialogue")
    session = ResearchSession(strategy=strategy)
    evidence = {}
    audit = []
    final = None
    for index, step in enumerate(steps):
        if type(step) is not dict or set(step) != {"kind", "payload"}:
            raise ValueError("invalid dialogue step")
        kind, payload = step["kind"], step["payload"]
        if kind == "tool":
            if final is not None:
                raise ValueError("tool request after final proposal")
            result = session.call(payload)
            for row in result["results"]:
                evidence[row["experiment_id"]] = row
            audit.append({"step": index, "kind": "tool", "tool": payload["tool"],
                          "request_sha256": hashlib.sha256(_encoded(payload)).hexdigest(),
                          "result_sha256": hashlib.sha256(_encoded(result)).hexdigest(),
                          "returned_ids": [row["experiment_id"] for row in result["results"]]})
        elif kind == "proposal":
            if final is not None or index != len(steps) - 1:
                raise ValueError("final proposal must terminate dialogue")
            final = validate_proposal(payload, evidence, strategy)
            audit.append({"step": index, "kind": "proposal",
                          "proposal_sha256": hashlib.sha256(_encoded(payload)).hexdigest()})
        else:
            raise ValueError("unsupported dialogue step")
        if len(_encoded(audit)) > MAX_TRANSCRIPT_BYTES:
            raise ValueError("audit budget exceeded")
    if final is None:
        raise ValueError("scripted session did not finish with a proposal")
    return {"result": final, "audit": audit, "tool_calls": session.calls,
            "evidence_bytes": session.used_bytes, "groq_calls": 0, "training_runs": 0}


class DisabledGroqAdapter:
    """Explicit placeholder. No credentials, HTTP library, or callable live path."""

    def complete(self, *args, **kwargs):
        raise RuntimeError("Live Groq integration is not implemented or authorized")


def offline_demonstration():
    return run_scripted_session(strategy="gpt-oss-120b", steps=[
        {"kind": "tool", "payload": {"tool": "search_experiments", "arguments": {"dataset": "ECG200"}}},
        {"kind": "tool", "payload": {"tool": "get_pareto_front", "arguments": {}}},
        {"kind": "proposal", "payload": {
            "hypothesis": "Width 10 may offer a useful parameter/accuracy trade-off.",
            "rationale": "Round 1 passed the gate while Round 2 failed; test this as a new hypothesis, not a predicted outcome.",
            "local_width": 10, "dataset": "ECG200", "epochs": 100,
            "evidence_ids": ["dev4-r1-gpt-oss-120b", "dev4-r2-gpt-oss-120b"],
            "expected_measurements": ["candidate_correct", "candidate_parameters", "gate_pass"]}},
    ])


if __name__ == "__main__":
    print(json.dumps(offline_demonstration(), indent=2, sort_keys=True))
