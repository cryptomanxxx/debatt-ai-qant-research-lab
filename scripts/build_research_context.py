"""Build a compact, provider-neutral evidence packet from verified results."""
import json
from pathlib import Path

results = Path("results")
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

packet = {
    "schema_version": 1,
    "purpose": "Evidence packet for the next falsifiable research proposal.",
    "rules": [
        "Use repository results as experimental evidence.",
        "Separate observations from hypotheses.",
        "Propose only; human approval is required before compute.",
        "Q.ANT CPU results do not establish photonic latency, energy, or hardware performance."
    ],
    "experiments": experiments
}
out.write_text(json.dumps(packet, indent=2) + "\n")
print("Built", out, "from", len(experiments), "result files")
