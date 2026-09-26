"""Build the AI Researcher evidence packet from historical and PNN-v1 evidence."""
import json
from pathlib import Path

out=Path("research_queue/context/latest.json")
out.parent.mkdir(parents=True,exist_ok=True)

def load_json(path):
 try:return json.loads(path.read_text())
 except Exception:return None

historical=[]
for path in sorted(Path("results").glob("exp*.json")):
 d=load_json(path)
 if not d: continue
 item={"file":path.name,"experiment_id":d.get("experiment_id"),"backend":d.get("backend"),"dataset":d.get("dataset")}
 for k in ("summary","pareto_front"):
  if d.get(k) is not None:item[k]=d[k]
 historical.append(item)

pnn=[]
for path in sorted(Path("pnn-v1/results").glob("result*.json")):
 d=load_json(path)
 if not d:continue
 pnn.append({
  "file":path.name,
  "experiment_id":d.get("experiment_id"),
  "proposal_id":d.get("proposal_id"),
  "timestamp_utc":d.get("timestamp_utc"),
  "backend":d.get("backend"),
  "dataset":d.get("dataset"),
  "configuration":d.get("configuration"),
  "success_criteria_met":d.get("success_criteria_met"),
  "decision":d.get("promotion_decision") or d.get("intervention_decision") or d.get("diagnostic_decision") or d.get("decision"),
  "selected_candidate":d.get("selected_candidate"),
  "summary":d.get("summary")
 })

historical_analyses=[]
for path in sorted(Path("research_queue/analyses").glob("*.json")):
 d=load_json(path)
 if d:
  historical_analyses.append({"file":path.name,"analysis_id":d.get("analysis_id"),"experiment_id":d.get("experiment_id"),"hypothesis_assessment":d.get("hypothesis_assessment"),"interpretation":d.get("interpretation"),"next_research_direction":d.get("next_research_direction")})

pnn_analyses=[]
for path in sorted(Path("pnn-v1/analyses").glob("analysis*.md")):
 pnn_analyses.append({"file":path.name,"text":path.read_text()})

# File names are not chronological (for example result_w8_confirmation sorts
# after result_w8_100epoch_confirmation). Use the recorded UTC timestamp.
pnn.sort(key=lambda x: x.get("timestamp_utc") or "")

human_review=load_json(Path("research_queue/human_review/latest.json"))

packet={
 "schema_version":6,
 "purpose":"Evidence packet for the next falsifiable research proposal.",
 "rules":[
  "Use repository results as experimental evidence.",
  "Separate observations from hypotheses.",
  "Propose only; human approval is required before compute.",
  "Q.ANT CPU/software simulation results do not establish photonic latency, energy, throughput, optical noise, or physical-hardware performance.",
  "For fixed-size classification gates, prefer aggregate integer correct counts over floating-point mean-accuracy boundaries."
 ],
 "active_pnn_v1_model":{
  "name":"Alpha5",
  "promoted_by":"PNN-v1-Exp020",
  "architecture":"96 → 6 contiguous windows×16 → 6× Q.ANT Fourier/KAN 16→16 → concat96 → 3 independent Q.ANT Fourier/KAN 96→2 readouts → arithmetic mean of logits",
  "frequencies":[1,2],
  "ecg200_parameter_count":8550,
  "model_file":"pnn-v1/models/alpha5.md"
 },
 "latest_human_review":human_review,
 "current_frontier":(
  {
   "latest_completed_experiment":pnn[-1].get("experiment_id") or pnn[-1].get("file"),
   "result_file":pnn[-1].get("file"),
   "success_criteria_met":pnn[-1].get("success_criteria_met"),
   "decision":pnn[-1].get("decision"),
   "configuration":pnn[-1].get("configuration"),
   "summary":pnn[-1].get("summary"),
   "research_instruction":"Treat this latest completed result as evidence. Do not reinterpret a failed preregistered gate as success. Formulate the next falsifiable research question from the evidence rather than repeating a completed experiment."
  }
  if pnn else
  {
   "latest_completed_experiment":None,
   "research_instruction":"No completed PNN-v1 experiment was found. Formulate the next falsifiable research question from available evidence."
  }
 ),
 "historical_toolkit_experiments":historical,
 "pnn_v1_experiments":pnn,
 "historical_researcher_analyses":historical_analyses,
 "pnn_v1_analyses":pnn_analyses
}
out.write_text(json.dumps(packet,indent=2)+"\n")
print("Built",out,"with",len(historical),"historical results and",len(pnn),"PNN-v1 results")
