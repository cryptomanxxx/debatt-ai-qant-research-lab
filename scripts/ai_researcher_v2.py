"""Groq-powered proposal-only AI Researcher v2.

Reads the repository evidence packet and writes a structured draft. It never
approves or executes compute. Human approval remains a separate workflow.
"""
import json, os, re, sys, urllib.request
from pathlib import Path

MODEL="openai/gpt-oss-120b"
CONTEXT=Path("research_queue/context/latest.json")
OUT=Path("research_queue/ai_researcher/latest.json")

def groq(messages):
    payload={
        "model":MODEL,
        "messages":messages,
        "temperature":0,
        "max_completion_tokens":3000,
    }
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
    with urllib.request.urlopen(req,timeout=120) as r:
        return json.load(r)["choices"][0]["message"]["content"]

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
    raw=groq([{"role":"system","content":system},{"role":"user","content":user}])
    try: data=json.loads(raw)
    except Exception:
        m=re.search(r"\{.*\}",raw,re.S)
        if not m: raise
        data=json.loads(m.group(0))
    required=("evidence_frontier","analysis","hypothesis","proposal")
    missing=[k for k in required if k not in data]
    if missing: raise SystemExit("AI Researcher output missing: "+", ".join(missing))
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
