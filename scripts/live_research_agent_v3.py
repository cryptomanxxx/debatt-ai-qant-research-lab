"""Opt-in Groq research orchestration; no training or workflow dispatch.

A client must be injected by the caller. CI uses a deterministic fake client.
The HTTP adapter requires an explicit enable flag and GROQ_API_KEY; importing
this module or running tests cannot send a live request.
"""
import hashlib
import json
import os
import urllib.request

from scripts.research_tool_calling_v3 import ResearchSession, ALLOWED_TOOLS
from scripts.scientific_reasoning_v3 import validate_proposal, _encoded
from scripts.research_memory_v3 import _strategy

MODEL = "openai/gpt-oss-120b"
MAX_MODEL_CALLS = 6
MAX_TOTAL_TOKENS = 12000
MAX_RESPONSE_BYTES = 8192
MAX_MESSAGES_BYTES = 32000
MAX_OUTPUT_TOKENS = 1024
MAX_INPUT_TOKENS_PER_CALL = 4000
PROPOSAL_SCHEMA_INSTRUCTIONS = (
    'For a proposal, output exactly {"kind":"proposal","payload":{'
    '"hypothesis":"testable explanation of 10 to 1000 characters",'
    '"rationale":"evidence-based rationale of 10 to 1000 characters",'
    '"local_width":10,"dataset":"ECG200","epochs":100,'
    '"evidence_ids":["dev4-r1-gpt-oss-120b"],'
    '"expected_measurements":["candidate_correct","candidate_parameters","gate_pass"]}}. '
    "The payload must contain exactly these seven fields, with no extras. "
    "local_width must be an integer from 4 to 15 (not a boolean); dataset must "
    "be ECG200 and epochs must be the integer 100. evidence_ids must be a list "
    "of 1 to 5 distinct IDs actually returned to you by this session's tools, "
    "all belonging to your assigned strategy. The three expected_measurements "
    "must appear exactly once each, with no additional measurements. "
    "The example evidence ID is illustrative only: never cite it unless it was "
    "actually returned by a tool in this session. "
)
SYSTEM = (
    "You are a scientific research assistant. You may request one read-only "
    "research tool at a time. Output ONLY one JSON object per turn. "
    'A tool request is {"kind":"tool","payload":{"tool":"search_experiments",'
    '"arguments":{"dataset":"ECG200"}}}; allowed tool names are '
    "search_experiments, get_experiment, and get_pareto_front. "
    + PROPOSAL_SCHEMA_INSTRUCTIONS +
    'Alternatively, abstain with {"kind":"insufficient_evidence",'
    '"payload":{"reason":"at least ten characters explaining the limitation"}}. '
    "Only cite experiment IDs actually returned by tools. "
    "Never request training, GitHub operations, or cross-strategy evidence. "
    "A proposal is not permission to execute an experiment."
)


class GroqHTTPAdapter:
    """Explicit opt-in HTTP adapter; no implicit retries or fallback models."""

    def __init__(self, *, enabled=False, api_key=None):
        if enabled is not True:
            raise RuntimeError("live Groq use requires explicit enabled=True")
        self._key = api_key or os.environ.get("GROQ_API_KEY")
        if not self._key:
            raise RuntimeError("GROQ_API_KEY is required")

    def complete(self, messages):
        payload = {"model": MODEL, "messages": messages, "temperature": 0,
                   "max_completion_tokens": MAX_OUTPUT_TOKENS,
                   "response_format": {"type": "json_object"}}
        request = urllib.request.Request(
            "https://api.groq.com/openai/v1/chat/completions",
            data=_encoded(payload),
            headers={"Authorization": "Bearer " + self._key,
                     "Content-Type": "application/json"},
            method="POST")
        with urllib.request.urlopen(request, timeout=45) as response:
            raw = response.read(MAX_RESPONSE_BYTES + 1)
        if len(raw) > MAX_RESPONSE_BYTES:
            raise ValueError("Groq response too large")
        result = json.loads(raw)
        if result["model"] != MODEL or len(result["choices"]) != 1:
            raise ValueError("unexpected model or choices")
        choice = result["choices"][0]
        if choice["finish_reason"] != "stop":
            raise ValueError("truncated or unfinished model response")
        usage = result["usage"]
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            if type(usage.get(key)) is not int or usage[key] < 0:
                raise ValueError("invalid Groq token usage")
        if usage["total_tokens"] != usage["prompt_tokens"] + usage["completion_tokens"]:
            raise ValueError("inconsistent token usage")
        if usage["prompt_tokens"] > MAX_INPUT_TOKENS_PER_CALL or usage["completion_tokens"] > MAX_OUTPUT_TOKENS:
            raise ValueError("per-call token budget exceeded")
        return {"content": choice["message"]["content"], "usage": usage}


def run_research(*, client, strategy="gpt-oss-120b", research_question,
                 max_model_calls=MAX_MODEL_CALLS, max_total_tokens=MAX_TOTAL_TOKENS):
    """Client-driven read-only dialogue. No external side effects beyond client.complete."""
    strategy = _strategy(strategy)
    if type(research_question) is not str or not 10 <= len(research_question) <= 500:
        raise ValueError("invalid research question")
    if type(max_model_calls) is not int or not 1 <= max_model_calls <= MAX_MODEL_CALLS:
        raise ValueError("invalid model-call budget")
    if type(max_total_tokens) is not int or not 1 <= max_total_tokens <= MAX_TOTAL_TOKENS:
        raise ValueError("invalid total token budget")
    session = ResearchSession(strategy=strategy)
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": "Strategy: " + strategy + "\nQuestion: " + research_question}]
    evidence, audit, total_tokens = {}, [], 0
    for index in range(max_model_calls):
        if len(_encoded(messages)) > MAX_MESSAGES_BYTES:
            raise ValueError("message context byte budget exceeded")
        response = client.complete(messages)
        if type(response) is not dict or set(response) != {"content", "usage"}:
            raise ValueError("invalid model response")
        content, usage = response["content"], response["usage"]
        if type(content) is not str or len(content.encode("utf-8")) > MAX_RESPONSE_BYTES:
            raise ValueError("invalid model content")
        if type(usage) is not dict or any(type(usage.get(k)) is not int or usage[k] < 0
                                         for k in ("prompt_tokens", "completion_tokens", "total_tokens")):
            raise ValueError("invalid token usage")
        if usage["total_tokens"] != usage["prompt_tokens"] + usage["completion_tokens"]:
            raise ValueError("inconsistent token usage")
        if usage["prompt_tokens"] > MAX_INPUT_TOKENS_PER_CALL or usage["completion_tokens"] > MAX_OUTPUT_TOKENS:
            raise ValueError("per-call token budget exceeded")
        total_tokens += usage["total_tokens"]
        if total_tokens > max_total_tokens:
            raise ValueError("total model token budget exceeded")
        step = json.loads(content)
        if type(step) is not dict or set(step) != {"kind", "payload"}:
            raise ValueError("invalid model step")
        kind, payload = step["kind"], step["payload"]
        entry = {"step": index, "kind": kind, "model_output_sha256": hashlib.sha256(content.encode()).hexdigest(),
                 "tokens": usage["total_tokens"]}
        if kind == "tool":
            result = session.call(payload)
            for row in result["results"]:
                evidence[row["experiment_id"]] = row
            entry["result_sha256"] = hashlib.sha256(_encoded(result)).hexdigest()
            entry["returned_ids"] = [row["experiment_id"] for row in result["results"]]
            messages.extend(({"role": "assistant", "content": content},
                             {"role": "user", "content": "Verified read-only tool result: " + json.dumps(result, sort_keys=True)}))
        elif kind == "proposal":
            validated = validate_proposal(payload, evidence, strategy)
            audit.append(entry)
            return {"result": validated, "audit": audit, "model_calls": index + 1,
                    "tool_calls": session.calls, "total_tokens": total_tokens, "training_runs": 0}
        elif kind == "insufficient_evidence":
            if type(payload) is not dict or set(payload) != {"reason"} or type(payload["reason"]) is not str or not 10 <= len(payload["reason"]) <= 1000:
                raise ValueError("invalid abstention reason")
            audit.append(entry)
            return {"result": {"schema_version": 1, "status": "insufficient_evidence",
                               "strategy": strategy, "reason": payload["reason"],
                               "requires_separate_human_compute_approval": True},
                    "audit": audit, "model_calls": index + 1, "tool_calls": session.calls,
                    "total_tokens": total_tokens, "training_runs": 0}
        else:
            raise ValueError("unapproved model action")
        audit.append(entry)
    raise ValueError("model-call budget exhausted without a validated conclusion")
