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
                    "test_shape":{"type":"array","items":{"type":"integer","enum":[100,96]},"minItems":2,"maxItems":2}
                },"required":["name","test_shape"],"additionalProperties":False},
                "seeds":{"type":"array","items":{"type":"integer","minimum":1},"minItems":3,"maxItems":10},
                "epochs":{"type":"integer","minimum":1,"maximum":200},
                "candidates":{"type":"array","items":{"type":"object","properties":{
                    "local_width":{"type":"integer","minimum":1}
                },"required":["local_width"],"additionalProperties":False},"minItems":1},
                "procedure":{"type":"array","items":{"type":"string"},"minItems":1}
            },"required":["dataset","seeds","epochs","candidates","procedure"],"additionalProperties":False},
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
        "recent_pnn_v1_experiments":full.get("pnn_v1_experiments",[])[-8:],
        "recent_pnn_v1_analyses":full.get("pnn_v1_analyses",[])[-3:],
    }
    system="""You are the proposal-only AI Researcher for Debatt-AI Q.ANT Research Lab.
Use only the supplied repository evidence. Separate observation from hypothesis.\nFor experiment history, treat structured fields (configuration, summary, decision,\nsuccess_criteria_met) as authoritative. Never infer or invent epochs, seeds, denominators,\nor outcomes from experiment names or prose when structured values are available.
Never claim simulation establishes physical photonic latency, energy, throughput,
optical noise, or hardware performance. Human approval is mandatory before compute.
Return one JSON object only with keys:
researcher, model, evidence_frontier, analysis, hypothesis, proposal.
proposal must contain: title, research_question, rationale, experiment_design,
success_criteria, requested_budget, risks, requires_human_approval.\nevidence_frontier, analysis, hypothesis, rationale, success_criteria, and risks must each be plain JSON strings, not objects or arrays.
requested_budget must contain positive integers: max_parameters, max_candidates,
max_epochs, max_train_samples, max_test_samples.
For fixed-size classification thresholds use aggregate integer correct counts,\nnot floating-point mean-accuracy boundary comparisons. Never propose an experiment\nthat simply repeats a completed candidate set. Check recent experiments for novelty.\nIf you specify N random seeds and M test samples, every aggregate-correct threshold\nmust be mathematically possible on N*M predictions and must state that denominator.\nA confirmation experiment must use a fresh preregistered seed set independent of the\ncompleted discovery experiment. Compare control and candidate on the SAME fresh seed\nset. A relative performance gate must compare against the concurrent control (for\nexample candidate_correct >= control_correct - 10), never against a historical absolute\ncount such as 887. Identify it explicitly as confirmation, not architecture search.
Exp021 already studied local-width compression and evaluated w8 and w12. Its structured
result is authoritative: PNN-v1-Exp021 used 10 seeds [11,22,33,44,55,66,77,88,99,111],
100 epochs, and 100 ECG200 test samples per seed (1000 predictions/model); aggregate
Q.ANT correct was w8=887, w12=884, and w16 control=897.
The later 50-epoch confirmation used seeds [121,131,141,151,161] and failed its gate:
w8=424/500 versus concurrent w16=441/500. The completed 100-epoch confirmation used
seeds [42,43,44,45,46] and PASSED its preregistered gate: w8=444/500 versus concurrent
w16=448/500, decision w8_boundary_confirmed.
Treat that successful 100-epoch confirmation as completed evidence. Do NOT propose the
same w8/w16 100-epoch experiment with seeds [42,43,44,45,46] again, and do not call a
rerun with identical seeds an independent replication. Advance to a genuinely new,
falsifiable research question. Because w8 is narrower than w12 and has already passed
its confirmation gate, a w12 experiment CANNOT establish that w12 is the minimal viable
local width. Do NOT call w8 the "minimal viable width" either: widths below 8 have not
been exhaustively tested under the same confirmation protocol. State only that w8 passed
the defined preregistered gate. If proposing w12, motivate it instead as an independent
robustness / width-response measurement (for example whether an intermediate width
behaves consistently under fresh seeds), and do not describe it as finding the minimum
width or as necessary to prove compression already demonstrated by w8.
For any experiment described as using fresh seeds, every seed must be absent from all
structured prior experiment configurations in the supplied context. In particular,
avoid the already used seeds [11,22,33,44,55,66,77,88,99,111],
[121,131,141,151,161], and [42,43,44,45,46].
For a 100-epoch confirmation proposal, requested_budget.max_epochs MUST be exactly 100,
not a looser ceiling such as 200. The procedure MUST explicitly state that candidate and
concurrent control are trained/evaluated on the same preregistered seeds, for 100 epochs,
using the same ECG200 train/test split (100 train and 100 test samples), and that results
are aggregated over 500 test predictions per model. Do not replace these protocol details
with abstract labels such as train_models or evaluate_qant_correct. Any true replication must use a fresh preregistered seed
set not used by the experiment being replicated. Keep the proposal bounded and use
structured seeds and epochs in experiment_design. PNN Experiment Engine v1 is a reviewed executable family for local-width confirmation
experiments. A proposal can use it without a new Python template when it uses ECG200,
exactly one candidate local_width from 4 through 15 plus the concurrent width-16 control,
exactly five fresh previously unused positive integer seeds, exactly 100 epochs, 100 train
and 100 test samples, 500 predictions per model, and the fixed relative gate
candidate.aggregate_qant_correct >= width16_control.aggregate_qant_correct - 10.
For this family set max_candidates=2, max_epochs=100, max_train_samples=100,
max_test_samples=100, and max_parameters between 8550 and 10000. Do not repeat a
completed design merely because the engine can execute it. A novel proposal outside this
reviewed family may still be scientifically proposed, but it must remain non-executable
until a new reviewed experiment family is added.
If latest_human_review contains a rejected proposal, treat its comment as mandatory\nreview feedback for the next proposal. Explicitly correct the rejected design rather\nthan repeating it. Human feedback is guidance only and can never authorize compute."""
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
