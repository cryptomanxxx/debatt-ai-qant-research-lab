"""Build the stable, web-facing research dashboard JSON."""
import json
from datetime import datetime, timezone
from pathlib import Path

RESULTS = Path("results")
OUT = Path("public/research-dashboard.json")

experiments = []
latest_candidates = []

EXPERIMENT_TYPES = {
    "exp001_baseline": "baseline",
    "exp002_width_search": "architecture_search",
    "exp003_architecture_search": "architecture_search",
    "exp004_depth_topology": "architecture_search",
    "exp005_auto_topology_search": "architecture_search",
    "exp006_multiseed_replication": "validation",
    "exp007_expansion_ratio": "architecture_search",
    "exp008_full_mnist_validation": "validation",
    "exp009_cross_dataset_validation": "validation",
    "exp010_qant_fourier_layer": "architecture_search",
    "exp011_qant_fourier_capacity": "architecture_search",
    "exp012_qant_fourier_robust_training": "training_method",
}

for p in sorted(RESULTS.glob("exp*.json")):
    try:
        o = json.loads(p.read_text())
    except Exception:
        continue

    experiment_id = o.get("experiment_id")
    experiment_type = EXPERIMENT_TYPES.get(experiment_id, "other")
    summary = o.get("summary") or {}
    item = {
        "id": experiment_id,
        "experiment_type": experiment_type,
        "completed": True,
        "result_count": len(o.get("results") or []),
        "summary_count": len(summary),
        "dataset": o.get("dataset"),
        "backend": o.get("backend"),
        "timestamp_utc": o.get("timestamp_utc") or o.get("timestamp"),
        "pareto_front": o.get("pareto_front", []),
        "pareto_applicable": experiment_type == "architecture_search",
    }
    cfg = o.get("configuration") or {}
    if cfg:
        item["configuration"] = {
            k: cfg[k]
            for k in ("train_samples", "test_samples", "epochs", "planned_runs")
            if k in cfg
        }
    experiments.append(item)

    if o.get("summary") and item["timestamp_utc"]:
        latest_candidates.append(o)

latest = max(
    latest_candidates,
    key=lambda o: o.get("timestamp_utc") or o.get("timestamp"),
) if latest_candidates else None

featured = []
if latest:
    for name, s in latest.get("summary", {}).items():
        featured.append({
            "architecture": name,
            "parameters": s["parameter_count"],
            "mean_accuracy": s["mean_qant_accuracy"],
            "std_accuracy": s.get("std_qant_accuracy"),
            "pareto": name in latest.get("pareto_front", []),
        })
    featured.sort(key=lambda x: x["parameters"])

payload = {
    "schema_version": 2,
    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    "project": {
        "name": "Debatt-AI Q.ANT Research Lab",
        "description": "Experimental architecture research using the Q.ANT Native Computing Toolkit.",
        "current_backend": "qant-cpu",
        "backend_disclaimer": (
            "Q.ANT CPU-backend results validate software execution and numerical behavior. "
            "They are not measurements of photonic hardware latency, energy efficiency, "
            "throughput, or optical behavior."
        ),
    },
    "status": {
        "completed_experiments": len(experiments),
        "latest_completed_experiment": latest.get("experiment_id") if latest else None,
        "current_research_direction": (
            ("Q.ANT-specific architecture research: investigating whether hardware-supported "
            "nonlinearities and KAN-style operations can produce more parameter-efficient "
            "neural architectures than conventional ReLU networks.")
        ),
    },
    "experiments": experiments,
    "featured_comparison": {
        "experiment_id": latest.get("experiment_id") if latest else None,
        "dataset": latest.get("dataset") if latest else None,
        "metric": "mean_qant_accuracy",
        "architectures": featured,
    },
    "research_loop": [
        "results",
        "AI researcher",
        "falsifiable hypothesis",
        "human approval",
        "guarded Q.ANT experiment",
        "new results",
    ],
    "provenance": {
        "source": "Generated from version-controlled result files in this repository.",
        "note": (
            "The public schema intentionally omits internal job, proposal, "
            "execution and governance details."
        ),
    },
}

OUT.parent.mkdir(exist_ok=True)
OUT.write_text(json.dumps(payload, indent=2) + "\n")
print(OUT)
