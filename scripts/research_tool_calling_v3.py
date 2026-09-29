"""Offline, fail-closed Research Tool Calling orchestration. No API or compute.

Tool requests are untrusted model-shaped data. The strategy is fixed by the
trusted caller, never accepted from a request. This module does not import Groq,
dispatch workflows, execute shell commands, or write research evidence.
"""
import json
from scripts import research_memory_v3 as memory

ALLOWED_TOOLS = frozenset(("search_experiments", "get_experiment", "get_pareto_front"))
MAX_CALLS = 5
MAX_SESSION_BYTES = 12000
MAX_REQUEST_BYTES = 2048
MAX_TOOL_RESULT_BYTES = 12000


class ResearchSession:
    def __init__(self, *, strategy, max_calls=MAX_CALLS, max_session_bytes=MAX_SESSION_BYTES):
        self.strategy = memory._strategy(strategy)
        if type(max_calls) is not int or not 1 <= max_calls <= MAX_CALLS:
            raise ValueError("invalid tool-call budget")
        if type(max_session_bytes) is not int or not 1 <= max_session_bytes <= MAX_SESSION_BYTES:
            raise ValueError("invalid session byte budget")
        self.max_calls = max_calls
        self.max_session_bytes = max_session_bytes
        self.calls = 0
        self.used_bytes = 0
        self.closed = False

    def call(self, request):
        """Execute exactly one validated, read-only tool request.

        The byte budget covers serialized returned evidence; callers must
        separately budget any system prompt, model reasoning and API envelope.
        A rejected request closes the session; there is no automatic retry.
        """
        if self.closed:
            raise ValueError("research session closed")
        try:
            if self.calls >= self.max_calls:
                raise ValueError("tool-call budget exhausted")
            if type(request) is not dict or set(request) != {"tool", "arguments"}:
                raise ValueError("invalid tool request shape")
            raw = json.dumps(request, sort_keys=True, allow_nan=False).encode("utf-8")
            if len(raw) > MAX_REQUEST_BYTES:
                raise ValueError("request byte budget exceeded")
            tool, args = request["tool"], request["arguments"]
            if type(tool) is not str or tool not in ALLOWED_TOOLS or type(args) is not dict:
                raise ValueError("unapproved tool or arguments")
            if "strategy" in args:
                raise ValueError("strategy is session-owned, not model-controlled")
            allowed = {
                "search_experiments": {"dataset", "local_width", "epochs", "limit"},
                "get_experiment": {"experiment_id"},
                "get_pareto_front": {"limit"},
            }[tool]
            if not set(args).issubset(allowed):
                raise ValueError("unapproved argument")
            if tool == "get_experiment" and set(args) != {"experiment_id"}:
                raise ValueError("experiment_id required")
            result = getattr(memory, tool)(strategy=self.strategy, **args)
            result_bytes = len(json.dumps(result, sort_keys=True, allow_nan=False).encode("utf-8"))
            if result_bytes > MAX_TOOL_RESULT_BYTES or self.used_bytes + result_bytes > self.max_session_bytes:
                raise ValueError("session evidence byte budget exceeded")
            self.calls += 1
            self.used_bytes += result_bytes
            return result
        except Exception:
            self.closed = True
            raise


def offline_demonstration():
    """Deterministic scripted requests, not a model invocation."""
    session = ResearchSession(strategy="gpt-oss-120b")
    search = session.call({"tool": "search_experiments", "arguments": {"dataset": "ECG200"}})
    front = session.call({"tool": "get_pareto_front", "arguments": {}})
    return {"search": search, "pareto": front, "tool_calls": session.calls,
            "evidence_bytes": session.used_bytes, "groq_calls": 0, "training_runs": 0}


if __name__ == "__main__":
    print(json.dumps(offline_demonstration(), sort_keys=True, indent=2))
