"""Fail-closed, evidence-bound second-pass scientific critique for AI Researcher v3.

The proposer and critic use separate model contexts. The critic cannot invoke
tools or execute experiments. This is another GPT-OSS pass, not independent
human peer review; its opinions are not empirically verified findings.
"""
import hashlib
import json

from scripts.live_research_agent_v3 import (
    MAX_INPUT_TOKENS_PER_CALL, MAX_MESSAGES_BYTES, MAX_MODEL_CALLS,
    MAX_OUTPUT_TOKENS, MAX_TOTAL_TOKENS, _checked_model_response, run_research,
)
from scripts.research_memory_v3 import _strategy
from scripts.scientific_reasoning_v3 import _encoded

# Reserve both one model turn and the maximum provider-reported per-call token
# allowance for critique. A completed proposal cannot use the review reserve.
RESEARCH_TOKEN_LIMIT = MAX_TOTAL_TOKENS - (
    MAX_INPUT_TOKENS_PER_CALL + MAX_OUTPUT_TOKENS)
REVIEW_FIELDS = frozenset((
    "assessment", "limitations", "alternative_explanations",
    "falsification_test", "reason", "evidence_ids",
))
REVIEW_ASSESSMENTS = frozenset((
    "testable_with_caveats", "revise_before_testing", "insufficient_evidence",
))
MAX_REVIEW_BYTES = 4096

REVIEW_SYSTEM = (
    "You are an adversarial SCIENTIFIC REVIEWER in a fresh, separate context, "
    "not the original proposing scientist. Use ONLY the verified evidence in "
    "this message. Treat the proposal as a hypothesis, never as an established "
    "result. Identify limitations, plausible alternative explanations (e.g. "
    "seed variation, sampling noise, non-monotonic width effects), and a "
    "concrete falsification test with measurable comparison against control. "
    "Two observed widths cannot establish a general trend or predict another "
    "width's accuracy. The provided automatic cautions are deterministic "
    "warnings that must be considered, not model-derived proof. Do not run or "
    "request experiments, tools, repository edits, or cross-strategy results. "
    "Output exactly ONE JSON object with this shape: "
    '{"kind":"critical_review","payload":{'
    '"assessment":"testable_with_caveats",'
    '"limitations":["one specific evidential limitation of 20 to 450 characters"],'
    '"alternative_explanations":["one plausible alternative of 20 to 450 characters"],'
    '"falsification_test":"a test of 20 to 600 characters",'
    '"reason":"an overall critique of 20 to 600 characters",'
    '"evidence_ids":["one experiment_id shown in verified evidence"]}}. '
    "The payload has exactly these six fields, no more. Limitations and "
    "alternative_explanations must each have 1 to 3 nonduplicate strings. "
    "assessment MUST be one of testable_with_caveats, revise_before_testing, "
    "or insufficient_evidence. Cite 1 to 5 distinct IDs actually shown in "
    "verified evidence, never an invented experiment or other strategy. "
    "testable_with_caveats means only that a hypothesis could be tested with "
    "separate human approval; it is not evidence that the hypothesis is true "
    "or permission for computation."
)


def automatic_cautions(proposal, observed):
    """Mechanically flag evidence scarcity and unobserved proposed width."""
    cautions = []
    if len(observed) < 3:
        cautions.append(
            "Only " + str(len(observed)) + " completed own-strategy observations "
            "were retrieved; this does not establish a reliable width/accuracy trend.")
    cited_count = len(proposal["evidence_ids"])
    if cited_count < 3:
        cautions.append(
            "The proposal cites only " + str(cited_count) +
            " distinct completed experiments; accuracy gains remain unproven.")
    if proposal["local_width"] not in {row["local_width"] for row in observed}:
        cautions.append(
            "Width " + str(proposal["local_width"]) +
            " has no completed observation in retrieved own-strategy history; "
            "any claimed improvement is an untested extrapolation.")
    return cautions


def _bounded_text(value, lower=20, upper=600):
    return type(value) is str and lower <= len(value.strip()) <= upper


def validate_critical_review(payload, *, proposal, observed, strategy):
    """Reject schema drift, unverifiable citations and ungrounded strategies."""
    strategy = _strategy(strategy)
    if type(payload) is not dict or set(payload) != REVIEW_FIELDS:
        raise ValueError("invalid critical review schema")
    if len(_encoded(payload)) > MAX_REVIEW_BYTES:
        raise ValueError("critical review too large")
    if payload["assessment"] not in REVIEW_ASSESSMENTS:
        raise ValueError("invalid critical review assessment")
    for name in ("limitations", "alternative_explanations"):
        items = payload[name]
        if type(items) is not list or not 1 <= len(items) <= 3 or any(
                not _bounded_text(x, 20, 450) for x in items):
            raise ValueError("invalid critical review " + name)
        if len(set(items)) != len(items):
            raise ValueError("duplicate critical review findings")
    for name in ("reason", "falsification_test"):
        if not _bounded_text(payload[name]):
            raise ValueError("invalid critical review " + name)
    ids = payload["evidence_ids"]
    if type(ids) is not list or not 1 <= len(ids) <= 5 or any(
            type(x) is not str for x in ids) or len(set(ids)) != len(ids):
        raise ValueError("invalid critical review evidence IDs")
    verified = {row["experiment_id"]: row for row in observed}
    if len(verified) != len(observed) or not set(ids).issubset(verified):
        raise ValueError("critical review cites unobserved evidence")
    if any(row["strategy"] != strategy for row in observed):
        raise ValueError("critical review sees cross-strategy evidence")
    return {
        "schema_version": 1,
        "review_type": "fresh_context_same_model",
        **payload,
        "automatic_cautions": automatic_cautions(proposal, observed),
        "reviewed_proposal_sha256": hashlib.sha256(_encoded(proposal)).hexdigest(),
        "requires_separate_human_compute_approval": True,
    }


def run_reviewed_research(*, client, research_question, strategy="gpt-oss-120b"):
    """One proposing dialogue plus a fresh-context critique, never training.

    The research stage has at most five model calls and 6976 reported tokens;
    the reviewer receives the sixth reserved call and up to 5024 tokens.
    A model abstention is returned without requesting a gratuitous critique.
    """
    strategy = _strategy(strategy)
    output = run_research(
        client=client, strategy=strategy, research_question=research_question,
        max_model_calls=MAX_MODEL_CALLS - 1, max_total_tokens=RESEARCH_TOKEN_LIMIT)
    if output["result"]["status"] == "insufficient_evidence":
        output["result"]["review_status"] = "not_applicable_no_proposal"
        output["review_model_calls"] = 0
        return output
    if output["result"]["status"] != "proposal_only":
        raise ValueError("unexpected proposing stage status")

    proposal = output["result"]["proposal"]
    observed = output["observed_evidence"]
    if type(observed) is not list or not observed:
        raise ValueError("review requires verified observed evidence")
    if any(row["strategy"] != strategy for row in observed):
        raise ValueError("review evidence strategy mismatch")
    cautions = automatic_cautions(proposal, observed)
    # Fresh messages: no original proposal dialogue, raw history or chain of thought.
    review_input = {
        "research_question": research_question,
        "proposal": proposal,
        "observed_verified_evidence": observed,
        "automatic_cautions": cautions,
    }
    messages = [
        {"role": "system", "content": REVIEW_SYSTEM},
        {"role": "user", "content": _encoded(review_input).decode("utf-8")},
    ]
    if len(_encoded(messages)) > MAX_MESSAGES_BYTES:
        raise ValueError("critical review context budget exceeded")
    content, usage = _checked_model_response(client.complete(messages))
    total_tokens = output["total_tokens"] + usage["total_tokens"]
    if total_tokens > MAX_TOTAL_TOKENS:
        raise ValueError("reviewed research total token budget exceeded")
    response = json.loads(content)
    if type(response) is not dict or set(response) != {"kind", "payload"} or (
            response["kind"] != "critical_review"):
        raise ValueError("invalid critical review model step")
    validated = validate_critical_review(
        response["payload"], proposal=proposal, observed=observed, strategy=strategy)
    output["result"]["critical_review"] = validated
    output["audit"].append({
        "step": output["model_calls"],
        "kind": "critical_review",
        "model_output_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        "reviewed_proposal_sha256": validated["reviewed_proposal_sha256"],
        "returned_ids": list(validated["evidence_ids"]),
        "tokens": usage["total_tokens"],
    })
    output["model_calls"] += 1
    output["review_model_calls"] = 1
    output["total_tokens"] = total_tokens
    return output
