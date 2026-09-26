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
    schema={"type":"object","properties":{
        "researcher":{"type":"string"},"model":{"type":"string"},"evidence_frontier":{"type":"string"},"analysis":{"type":"string"},"hypothesis":{"type":"string"},
        "proposal":{"type":"object","properties":{
            "title":{"type":"string"},"research_question":{"type":"string"},"rationale":{"type":"string"},
            "experiment_design":{"type":"object","properties":{
                "dataset":{"type":"object","properties":{
                    "name":{"type":"string","const":"ECG200"},
                    "test_shape":{"type":"array","items":{"type":"integer","minimum":1},"minItems":1}
                },"required":["name","test_shape"],"additionalProperties":False},
                "candidates":{"type":"array","items":{"type":"object","properties":{
                    "local_width":{"type":"integer","minimum":1}
                },"required":["local_width"],"additionalProperties":False},"minItems":1},
                "procedure":{"type":"array","items":{"type":"string"},"minItems":1}
            },"required":["dataset","candidates","procedure"],"additionalProperties":False},
            "success_criteria":{"type":"string"},
            "requested_budget":{"type":"object","properties":{
                "max_parameters":{"type":"integer","minimum":1},"max_candidates":{"type":"integer","minimum":1},
                "max_epochs":{"type":"integer","minimum":1},"max_train_samples":{"type":"integer","minimum":1},"max_test_samples":{"type":"integer","minimum":1}},
                "required":["max_parameters","max_candidates","max_epochs","max_train_samples","max_test_samples"],"additionalProperties":False},
            "risks":{"type":"string"},"requires_human_approval":{"type":"boolean","const":True}},
            "required":["title","research_question","rationale","experiment_design","success_criteria","requested_budget","risks","requires_human_approval"],
            "additionalProperties":False}},
        "required":["researcher","model","evidence_frontier","analysis","hypothesis","proposal"],"additionalProperties":False}
    payload={"model":MODEL,"messages":messages,"temperature":0,"max_completion_tokens":3000,
             "response_format":{"type":"json_schema","json_schema":{"name":"ai_researcher_proposal","strict":True,"schema":schema}}}
    for attempt in range(3):
        req=urllib.request.Request("https://api.groq.com/openai/v1/chat/completions",data=json.dumps(payload).encode(),
            headers={"Authorization":f"Bearer {os.environ['GROQ_API_KEY']}","Content-Type":"application/json","Accept":"application/json",
                     "User-Agent":"debatt-ai-qant-research-lab/2.0"},method="POST")
        try:
            with urllib.request.urlopen(req,timeout=120) as r:
                return json.load(r)["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as exc:
            if exc.code == 400:
                # Groq returns schema-validation details in the response body.
                # Print that diagnostic only; request headers/API key are never logged.
                try:
                    detail=exc.read().decode("utf-8","replace")
                except Exception:
                    detail="<unable to read Groq error body>"
                print(f"Groq HTTP 400 response: {detail}",file=sys.stderr)
                raise
            if exc.code != 429 or attempt == 2:
                raise
            try: delay=max(1.0,float(exc.headers.get("Retry-After")))
            except (TypeError,ValueError): delay=2.0*(2**attempt)
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
        "latest_human_review":full.get("latest_human_review"),
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
success_criteria, requested_budget, risks, requires_human_approval.\nevidence_frontier, analysis, hypothesis, rationale, success_criteria, and risks must each be plain JSON strings, not objects or arrays.
requested_budget must contain positive integers: max_parameters, max_candidates,
max_epochs, max_train_samples, max_test_samples.
For fixed-size classification thresholds use aggregate integer correct counts,\nnot floating-point mean-accuracy boundary comparisons. Never propose an experiment\nthat simply repeats a completed candidate set. Check recent experiments for novelty.\nIf you specify N random seeds and M test samples, every aggregate-correct threshold\nmust be mathematically possible on N*M predictions and must state that denominator.\nA confirmation experiment must use a fresh preregistered seed set independent of the\ncompleted discovery experiment. Compare control and candidate on the SAME fresh seed\nset. A relative performance gate must compare against the concurrent control (for\nexample candidate_correct >= control_correct - 10), never against a historical absolute\ncount such as 887. Identify it explicitly as confirmation, not architecture search.
Exp021 already studied local-width compression and evaluated w8 and w12. The next
confirmation must test ONLY the w8 boundary against the Alpha5 w16 control. Do not
introduce a new width such as w4 and do not repeat w12. Do not claim that local width
has never been varied; Exp021 is evidence that it has.\nThe latest completed w8 confirmation already tested 50 epochs and failed its relative\ngate: w8 aggregate Q.ANT correct was 424/500 versus concurrent w16 control 441/500.\nDo NOT repeat a 50-epoch w8 confirmation. The next proposed confirmation must test\nwhether longer training closes that observed gap: use EXACTLY preregistered seeds\n[42,43,44,45,46], train BOTH Alpha5 w8 and the concurrent Alpha5 w16 control for\nEXACTLY 100 epochs on the ECG200 TRAIN split (100 samples), evaluate both on the\nECG200 TEST split (100 samples), and aggregate exactly 500 predictions per model.\nIn experiment_design.candidates list ONLY local_width=8; the w16 model is the\nconcurrent control described in procedure, not a second candidate. Use requested\nbudget max_parameters=9000, max_candidates=1, max_epochs=100,\nmax_train_samples=100, max_test_samples=100. The success criterion must be the\nrelative aggregate gate candidate_correct >= control_correct - 10 on 5*100=500\npredictions. Do not substitute other seeds, epochs, widths, or an absolute threshold.\nIf latest_human_review contains a rejected proposal, treat its comment as mandatory\nreview feedback for the next proposal. Explicitly correct the rejected design rather\nthan repeating it. Human feedback is guidance only and can never authorize compute."""
    user="Repository research context:\n"+json.dumps(context,ensure_ascii=False)
    messages=[{"role":"system","content":system},{"role":"user","content":user}]
    raw=groq(messages)
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
