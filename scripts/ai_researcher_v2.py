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
        "response_format":{"type":"json_schema","json_schema":{"name":"ai_researcher_proposal","strict":True,"schema":schema}}
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

def main():
    if not os.environ.get("GROQ_API_KEY"): raise SystemExit("GROQ_API_KEY missing")
    if not CONTEXT.exists(): raise SystemExit(f"Missing {CONTEXT}")
    full=json.loads(CONTEXT.read_text())
    # Keep the model input focused and bounded. Full evidence remains in GitHub.
    context={
        "schema_version":full.get("schema_version"),
        "rules":full.get("rules",[]),
        "active_pnn_v1_model":full.get("active_pnn_v1_model"),
        "current_frontier":full.get("current_frontier"),
        "recent_pnn_v1_experiments":full.get("pnn_v1_experiments",[])[-6:],
        "recent_pnn_v1_analyses":full.get("pnn_v1_analyses",[])[-3:],
    }
    system="""You are the proposal-only AI Researcher for Debatt-AI Q.ANT Research Lab.
Use only the supplied repository evidence. Separate observation from hypothesis.
Never claim simulation establishes physical photonic latency, energy, throughput,
optical noise, or hardware performance. Human approval is mandatory before compute.
Return one JSON object only with keys:
researcher, model, evidence_frontier, analysis, hypothesis, proposal.
proposal must contain: title, research_question, rationale, experiment_design,
success_criteria, requested_budget, risks, requires_human_approval.
requested_budget must contain positive integers: max_parameters, max_candidates,
max_epochs, max_train_samples, max_test_samples.
For fixed-size classification thresholds use aggregate integer correct counts,\nnot floating-point mean-accuracy boundary comparisons. Never propose an experiment\nthat simply repeats a completed candidate set. Check recent experiments for novelty.\nIf you specify N random seeds and M test samples, every aggregate-correct threshold\nmust be mathematically possible on N*M predictions and must state that denominator.\nA confirmation experiment must use an independent preregistered seed set and must be\nidentified explicitly as confirmation rather than a new architecture search."""
    user="Repository research context:\n"+json.dumps(context,ensure_ascii=False)
    messages=[{"role":"system","content":system},{"role":"user","content":user}]
    raw=groq(messages)

    def parse_json(text):
        try:
            return json.loads(text)
        except Exception:
            m=re.search(r"\{.*\}",text,re.S)
            if not m:
                raise
            return json.loads(m.group(0))

    # If the model returns malformed JSON, do not heuristically rewrite the
    # research content in Python. Give it one bounded opportunity to serialize
    # the same proposal as strict JSON; fail closed if that also cannot parse.
    try:
        data=parse_json(raw)
    except (json.JSONDecodeError, ValueError):
        repair=(
            "Your previous response is not valid JSON. Return the COMPLETE response "
            "again as one strict JSON object using double-quoted property names and "
            "valid JSON syntax. Do not return a patch, markdown, code fences, or "
            "commentary. Preserve the research claims; this request is only to repair "
            "serialization. The required top-level structure is: researcher, model, "
            "evidence_frontier, analysis, hypothesis, proposal.\n\nPrevious response:\n"
            + raw
        )
        raw=groq(messages+[{"role":"assistant","content":raw},{"role":"user","content":repair}])
        try:
            data=parse_json(raw)
        except (json.JSONDecodeError, ValueError) as exc:
            raise SystemExit("AI Researcher returned invalid JSON after repair attempt") from exc

    required=("evidence_frontier","analysis","hypothesis","proposal")
    missing=[k for k in required if k not in data]

    # Models can occasionally return a structurally valid JSON object while
    # omitting one required top-level field. Give the researcher one bounded
    # repair attempt instead of publishing or silently inventing that field.
    if missing:
        repair=(
            "Your previous JSON object omitted required top-level key(s): "
            + ", ".join(missing)
            + ". Return the COMPLETE corrected JSON object, not a patch. "
              "It must contain exactly the requested top-level structure: "
              "researcher, model, evidence_frontier, analysis, hypothesis, proposal. "
              "Preserve claims only when supported by the repository context. "
              "Do not add markdown or commentary.\n\nPrevious response:\n"
            + raw
        )
        raw=groq(messages+[{"role":"assistant","content":raw},{"role":"user","content":repair}])
        data=parse_json(raw)
        missing=[k for k in required if k not in data]

    if missing:
        raise SystemExit("AI Researcher output missing after repair attempt: "+", ".join(missing))
    p=data["proposal"]
    for k in ("title","research_question","rationale","experiment_design","success_criteria","requested_budget","risks","requires_human_approval"):
        if k not in p: raise SystemExit("Proposal missing: "+k)
    if p["requires_human_approval"] is not True: raise SystemExit("AI proposal must require human approval")
    for k in ("max_parameters","max_candidates","max_epochs","max_train_samples","max_test_samples"):
        v=p["requested_budget"].get(k)
        if not isinstance(v,int) or isinstance(v,bool) or v<=0: raise SystemExit("Invalid requested_budget."+k)
    data["researcher"]="AI Researcher v2"
    data["model"]=MODEL
    data["status"]="draft_requires_human_review"
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(data,indent=2,ensure_ascii=False)+"\n")
    print("AI Researcher v2 wrote",OUT)
    print("Hypothesis:",data["hypothesis"])
    print("Proposal:",p["title"])

if __name__=="__main__": main()    raw=groq(messages)
    try:
        data=json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SystemExit("Groq strict structured output returned invalid JSON") from exc

    required=("evidence_frontier","analysis","hypothesis","proposal")
    missing=[k for k in required if k not in data]
    if missing:
        raise SystemExit("AI Researcher strict output missing: "+", ".join(missing))
    p=data["proposal"]
    for k in ("title","research_question","rationale","experiment_design","success_criteria","requested_budget","risks","requires_human_approval"):
        if k not in p: raise SystemExit("Proposal missing: "+k)
    if p["requires_human_approval"] is not True: raise SystemExit("AI proposal must require human approval")
    for k in ("max_parameters","max_candidates","max_epochs","max_train_samples","max_test_samples"):
        v=p["requested_budget"].get(k)
        if not isinstance(v,int) or isinstance(v,bool) or v<=0: raise SystemExit("Invalid requested_budget."+k)
    data["researcher"]="AI Researcher v2"
    data["model"]=MODEL
    data["status"]="draft_requires_human_review"
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(data,indent=2,ensure_ascii=False)+"\n")
    print("AI Researcher v2 wrote",OUT)
    print("Hypothesis:",data["hypothesis"])
    print("Proposal:",p["title"])

if __name__=="__main__": main()
