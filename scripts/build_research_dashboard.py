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
    "exp013_qant_fourier_mismatch_diagnostic": "diagnostic",
    "exp014_qant_fourier_error_decomposition": "diagnostic",
    "exp015_qant_aligned_training_surrogate": "training_method",
    "exp016_qant_activation_error_occupancy": "diagnostic",
    "exp017_qant_logit_margin_diagnostic": "diagnostic",
    "exp018_qant_frequency_controlled_g8": "architecture_search",
    "exp019_qant_amplitude_error_accumulation": "diagnostic",
    "exp020_qant_kan_additivity_ladder": "diagnostic",
    "exp021_qant_accumulation_scaling_law": "diagnostic",
    "exp022_qant_grouped_accumulation_architecture": "architecture_search",
    "exp023_qant_hierarchical_low_fanin": "architecture_search",
    "exp024_qant_hierarchical_architecture_search": "architecture_search",
    "exp025_qant_hierarchical_error_localization": "diagnostic",
    "exp026_qant_predictive_compatibility_model": "diagnostic",
    "exp027_qant_factorial_accumulation_map": "diagnostic",
}

for p in sorted(RESULTS.glob("exp*.json")):
    try:
        o = json.loads(p.read_text())
    except Exception:
        continue

    experiment_id = o.get("experiment_id")
    experiment_type = EXPERIMENT_TYPES.get(experiment_id, "other")
    summary = o.get("summary") or {}
    results = o.get("results")
    if isinstance(results, list):
        result_count = len(results)
    elif experiment_type == "baseline" and o.get("metrics"):
        # Legacy baseline files store one completed run in metrics instead of results[].
        result_count = 1
    else:
        result_count = 0
    pareto_front = o.get("pareto_front")
    if pareto_front is None and experiment_id == "exp002_width_search" and isinstance(results, list):
        # Legacy Exp002 predates explicit Pareto metadata. Every successive width
        # increased both parameter count and Q.ANT accuracy, so all six points
        # are non-dominated for the two objectives used by later searches.
        pareto_front = [f"h{r['hidden_units']}" for r in results]
    if pareto_front is None:
        pareto_front = []

    item = {
        "id": experiment_id,
        "experiment_type": experiment_type,
        "completed": True,
        "result_count": result_count,
        "summary_count": len(summary),
        "dataset": o.get("dataset"),
        "backend": o.get("backend"),
        "timestamp_utc": o.get("timestamp_utc") or o.get("timestamp"),
        "pareto_front": pareto_front,
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

    # The featured accuracy-vs-parameters chart is an architecture comparison,
    # so only architecture-search experiments are eligible to replace it.
    if (
        experiment_type == "architecture_search"
        and o.get("summary")
        and item["timestamp_utc"]
    ):
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
        "latest_completed_experiment": (
            max(
                (e for e in experiments if e.get("timestamp_utc")),
                key=lambda e: e["timestamp_utc"],
            )["id"]
            if any(e.get("timestamp_utc") for e in experiments)
            else None
        ),
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
