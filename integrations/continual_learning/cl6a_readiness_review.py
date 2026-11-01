"""CL-6A: strict read-only preregistration review. NEVER trains or generates.

The read-only validator checks a proposed study design; it cannot approve
science, generate/alter experimental data or authorize future computation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

PATH = Path(__file__).with_name("cl6a_readiness_protocol.json")
GATES = (
    "source_and_per_seed_data_digests",
    "causal_arm_equivalence",
    "paired_statistics_and_quality",
    "observed_event_scope_and_overhead",
    "train_only_learnability_calibration",
    "physical_energy_boundary_or_NOT_MEASURED",
    "bounded_runner_and_immutable_artifacts",
    "independent_scientific_review",
    "separate_explicit_marcus_training_approval",
)
ARMS = (
    ("dense_sgd", "global_sgd", "dense", False),
    ("local_hybrid", "local", "dense", False),
    ("sparse_local_hybrid", "local", "sparse", False),
    ("sparse_shared_index", "local", "sparse", True),
)
COMPARISONS = (
    ("primary_sparse", "local_hybrid", "sparse_local_hybrid"),
    ("secondary_index_reuse", "sparse_local_hybrid", "sparse_shared_index"),
    ("context_learning_rule", "dense_sgd", "local_hybrid"),
)


def review(protocol: dict) -> dict:
    if not isinstance(protocol, dict):
        raise ValueError("CL-6A expects a protocol object")
    if (protocol.get("kind") != "CL6A_CONTINUAL_LEARNING_READINESS_DRAFT"
            or protocol.get("version") != "cl6a-v1"
            or protocol.get("decision") != "NO_GO"):
        raise ValueError("unapproved or wrong-version CL-6A protocol")
    for key in ("scientific_approved", "model_training_authorized",
                "dataset_generation_authorized", "model_inference_authorized",
                "physical_energy_measurement_authorized", "new_results_present"):
        if protocol.get(key) is not False:
            raise ValueError(f"{key} must remain explicitly false")
    model = protocol.get("model", {})
    if model != {"shape": [16, 8, 2], "parameters_per_arm": 154}:
        raise ValueError("model shape/capacity changed")
    arms = protocol.get("arms", [])
    if not isinstance(arms, list) or len(arms) != 4:
        raise ValueError("exactly four arms required")
    if [(a.get("id"), a.get("learning"), a.get("forward"),
         a.get("active_index_reuse")) for a in arms] != list(ARMS):
        raise ValueError("arm identity or causal design changed")
    comparisons = protocol.get("comparisons", [])
    if not isinstance(comparisons, list) or len(comparisons) != 3:
        raise ValueError("comparison plan changed")
    if [(c.get("id"), c.get("left"), c.get("right"))
        for c in comparisons] != list(COMPARISONS):
        raise ValueError("primary/secondary comparison identity changed")
    design = protocol.get("future_design", {})
    expected_design = {
        "seeds": [101, 102, 103, 104, 105],
        "tasks": ["T1", "T2", "T3", "T4"],
        "train_records_per_task": 128,
        "test_records_per_task": 64,
        "maximum_updates_per_arm_seed": 512,
        "passes_per_task": 1,
        "replay": False,
        "task_id_feature": False,
        "test_tuning_allowed": False,
        "same_data_order_per_seed": True,
        "checkpoints": ["pretrain", "post_T1", "post_T2", "post_T3", "post_T4"],
    }
    if design != expected_design:
        raise ValueError("future design bounds or heldout rules changed")
    quality = protocol.get("preregistered_quality", {})
    if (type(quality.get("final_mean_accuracy_floor")) not in (int, float)
            or type(quality.get("paired_accuracy_difference_max_abs")) not in (int, float)
            or quality["final_mean_accuracy_floor"] != 0.65
            or quality["paired_accuracy_difference_max_abs"] != 0.05):
        raise ValueError("quality thresholds must remain preregistered")
    measurements = protocol.get("measurement_scope", {})
    for key in ("hardware_energy", "joules_per_step", "photonic_hardware"):
        if measurements.get(key) != "NOT_MEASURED":
            raise ValueError("unreviewed physical energy or photonic claim")
    for key in ("heldout_all_tasks_each_checkpoint", "balanced_accuracy",
                "forgetting_per_prior_task", "training_and_inference_separate",
                "logical_reads_and_writes", "active_index_visits",
                "nonlinearities_and_control", "instrumentation_overhead"):
        if measurements.get(key) is not True:
            raise ValueError(f"required metric missing: {key}")
    provenance = protocol.get("data_provenance", {})
    if (provenance.get("existing_single_seed_source_run") != 41311514598
            or provenance.get("existing_seed101_aggregate_only") is not True
            or provenance.get("per_seed_data_digests_frozen") is not False
            or provenance.get("source_and_runner_pinned_for_training") is not False
            or provenance.get("any_new_generation_requires_separate_approval") is not True):
        raise ValueError("provenance or generation authorization falsely asserted")
    analysis = protocol.get("analysis", {})
    if (analysis.get("paired_seed_descriptive_only") is not True
            or analysis.get("significance_claims") is not False
            or analysis.get("posthoc_exclusion_allowed") is not False
            or analysis.get("quality_conditioned_cost_interpretation") is not True):
        raise ValueError("unapproved statistical interpretation")
    if protocol.get("open_gates") != list(GATES):
        raise ValueError("all nine open gates must remain open")
    execution = protocol.get("execution_contract", {})
    if execution != {
        "workflow_dispatch_training": False,
        "training_module_import": False,
        "auto_gate_closure": False,
        "training_go_live": False,
    }:
        raise ValueError("no execution or automatic gate closure is authorized")
    if "NOT class-incremental" not in protocol.get("construct_validity", ""):
        raise ValueError("distribution shift must not be called class-incremental learning")
    return {
        "decision": "NO_GO",
        "training_authorized": False,
        "new_data_authorized": False,
        "architectures": len(arms),
        "planned_paired_seeds": len(design["seeds"]),
        "open_gates": list(GATES),
        "closed_gates": 0,
        "actual_experiment_results": False,
        "energy_measured": False,
    }


def summary(result: dict) -> str:
    lines = [
        "## CL-6A — continual learning experiment readiness (NO TRAINING)",
        "",
        "> **NO_GO**. This is a review of a future protocol, not a study execution. "
        "No model, dataset, inference, training or energy measurement was run.",
        "",
        "- Four planned arms: dense SGD, dense local, sparse local, shared-index sparse local.",
        "- Primary causal ablation: dense-local vs sparse-local; secondary: sparse-local vs shared-index.",
        "- Five proposed paired seeds (101–105), four tasks; 16→8→2 MLP; "
        "up to 512 updates per arm/seed, **not executed**.",
        "- Predeclared interpretation: final accuracy ≥0.65 and paired gap ≤0.05 "
        "per seed. Failure blocks source-cost advantages at matched quality.",
        "- XOR with activity shift is **not** class-incremental catastrophic-forgetting evidence.",
        "- Hardware energy, joules and photonic performance: **NOT_MEASURED**.",
        "",
        f"### Open prerequisites ({len(result['open_gates'])}; none closed)",
        "",
    ]
    lines.extend(f"- [ ] `{gate}`" for gate in result["open_gates"])
    lines += [
        "",
        "**Codex review, green CI, or merging does NOT give scientific sign-off "
        "or permission to train. Marcus must separately explicitly approve "
        "an exact, independently reviewed runner and protocol.**",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=PATH)
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args(argv)
    result = review(json.loads(args.protocol.read_text(encoding="utf-8")))
    rendered = summary(result)
    if args.summary is not None:
        with args.summary.open("a", encoding="utf-8") as out:
            out.write(rendered)
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
