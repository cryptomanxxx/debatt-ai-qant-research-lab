"""Build a compact, provider-neutral evidence packet from verified results and analyses."""
import json
from pathlib import Path

results = Path("results")
analyses_dir = Path("research_queue/analyses")
out = Path("research_queue/context/latest.json")
out.parent.mkdir(parents=True, exist_ok=True)

experiments = []
for path in sorted(results.glob("exp*.json")):
    data = json.loads(path.read_text())
    item = {"file": path.name, "experiment_id": data.get("experiment_id"),
            "backend": data.get("backend"), "dataset": data.get("dataset")}
    if data.get("summary") is not None:
        item["summary"] = data["summary"]
    if data.get("pareto_front") is not None:
        item["pareto_front"] = data["pareto_front"]
    experiments.append(item)

analyses = []
if analyses_dir.exists():
    for path in sorted(analyses_dir.glob("*.json")):
        data = json.loads(path.read_text())
        analyses.append({
            "file": path.name,
            "analysis_id": data.get("analysis_id"),
            "experiment_id": data.get("experiment_id"),
            "researcher": data.get("researcher"),
            "hypothesis_assessment": data.get("hypothesis_assessment"),
            "interpretation": data.get("interpretation"),
            "next_research_direction": data.get("next_research_direction")
        })

packet = {
    "schema_version": 2,
    "purpose": "Evidence packet for the next falsifiable research proposal.",
    "rules": [
        "Use repository results as experimental evidence.",
        "Separate observations from hypotheses.",
        "Propose only; human approval is required before compute.",
        "Q.ANT CPU results do not establish photonic latency, energy, or hardware performance."
    ],
    "experiments": experiments,
    "researcher_analyses": analyses
}
out.write_text(json.dumps(packet, indent=2) + "\n")
print("Built", out, "from", len(experiments), "results and", len(analyses), "analyses")
