"""Groq-powered proposal-only AI Researcher v2.

Reads the repository evidence packet and writes a structured draft. It never
approves or executes compute. Human approval remains a separate workflow.
"""
import json, os, sys, time, urllib.error, urllib.request
from pathlib import Path

MODEL="openai/gpt-oss-120b"
CONTEXT=Path("research_queue/context/latest.json")
OUT=Path("research_queue/ai_researcher/latest.json")

def groq(messages):
    schema={
        "type":"object",
        "properties":{
            "researcher":{"type":"string"},
            "model":{"type":"string"},
            "evidence_frontier":{},
            "analysis":{},
            "hypothesis":{},
            "proposal":{
                "type":"object",
                "properties":{
                    "title":{"type":"string"},
                    "research_question":{"type":"string"},
                    "rationale":{},
                    "experiment_design":{},
                    "success_criteria":{},
                    "requested_budget":{
                        "type":"object",
                        "properties":{
                            "max_parameters":{"type":"integer","minimum":1},
                            "max_candidates":{"type":"integer","minimum":1},
                            "max_epochs":{"type":"integer","minimum":1},
                            "max_train_samples":{"type":"integer","minimum":1},
                            "max_test_samples":{"type":"integer","minimum":1}
                        },
                        "required":["max_parameters","max_candidates","max_epochs","max_train_samples","max_test_samples"],
                        "additionalProperties":False
                    },
                    "risks":{},
                    "requires_human_approval":{"type":"boolean","const":True}
                },
                "required":["title","research_question","rationale","experiment_design","success_criteria","requested_budget","risks","requires_human_approval"],
                "additionalProperties":False
            }
        },
        "required":["researcher","model","evidence_frontier","analysis","hypothesis","proposal"],
        "additionalProperties":False
    }
    payload={
        "model":MODEL,
        "messages":messages,
        "temperature":0,
        "max_completion_tokens":3000,
        "response_format":{
            "type":"json_schema",
            "json_schema":{
                "name":"ai_researcher_proposal",
                "strict":True,
                "schema":schema
            }
        }
    }
    for attempt in range(3):
        req=urllib.request.Request(
            "https://api.groq.com/openai/v1/chat/completions",
            data=json.dumps(payload).encode(),
            headers={
                "Authorization":f"Bearer {os.environ['GROQ_API_KEY']}",
                "Content-Type":"application/json",
                "Accept":"application/json",
                "User-Agent":"debatt-ai-qant-research-lab/2.0",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req,timeout=120) as r:
                return json.load(r)["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as exc:
            if exc.code != 429 or attempt == 2:
                raise
            retry_after=exc.headers.get("Retry-After")
            try:
                delay=max(1.0,float(retry_after))
            except (TypeError,ValueError):
                delay=2.0*(2**attempt)
            print(f"Groq rate limited request; retrying in {delay:g}s",file=sys.stderr)
            time.sleep(delay)
    raw=groq(messages)
    try:
        data=json.loads(raw)
    except json.JSONDecodeError as exc:
        # strict Structured Outputs should make this unreachable; fail closed.
        raise SystemExit("Groq strict structured output returned invalid JSON") from exc

    required=("evidence_frontier","analysis","hypothesis","proposal")
    missing=[k for k in required if k not in data]
    if missing:
        raise SystemExit("AI Researcher strict output missing: "+", ".join(missing))

