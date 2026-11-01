"""CL-3W bounded H1 evidence review based on static CL-3T/U/V records.

Produces no scientific experiment, hardware estimate, or compute authorization.
"""
from __future__ import annotations

from integrations.continual_learning.cl3r_visualization import build_preview
from integrations.continual_learning.cl3t_crossover_analysis import analyze_crossover
from integrations.continual_learning.cl3u_sensitivity import sensitivity, check_reference
from integrations.continual_learning.cl3v_cost_components import component_report


def h1_evidence_review() -> dict:
    """Fail closed if the previously reviewed static proxies disagree."""
    check_reference()
    preview = build_preview()
    cross = analyze_crossover(preview)
    shape_sweep = sensitivity()
    components = component_report()
    lookup = {(r["phase"], r["arm"], r["active"]): r for r in components["rows"]}
    for point in preview["series"]:
        active = point["active_input_bits"]
        for arm, values in point["by_arm"].items():
            if lookup[("training", arm, active)]["total_bytes"] != values["training_logical_payload_bytes_per_assumed_update"]:
                raise ValueError("CL-3V training evidence diverges from CL-3R")
            if lookup[("inference", arm, active)]["total_bytes"] != values["inference_logical_payload_bytes_per_assumed_prediction"]:
                raise ValueError("CL-3V inference evidence diverges from CL-3R")
    comparisons = {(r["phase"], r["arm"]): r for r in cross["comparisons"]}
    sparse_inference = comparisons[("inference", "sparse_local_hybrid")]
    if (sparse_inference["last_strictly_better_activity"],
            sparse_inference["first_not_better_activity"]) != (13, 14):
        raise ValueError("reference breakpoint changed: human review required")
    dense2 = lookup[("inference", "dense_sgd", 2)]["total_bytes"]
    sparse2 = lookup[("inference", "sparse_local_hybrid", 2)]["total_bytes"]
    dense12 = lookup[("inference", "dense_sgd", 12)]["total_bytes"]
    sparse12 = lookup[("inference", "sparse_local_hybrid", 12)]["total_bytes"]
    if (dense2, sparse2, dense12, sparse12) != (1376, 404, 1376, 1224):
        raise ValueError("reference totals changed: human review required")
    if len(shape_sweep["scenarios"]) != 9 or len(components["rows"]) != 136:
        raise ValueError("incomplete evidence sweep")
    return {
        "kind": "CL3W_H1_STATIC_EVIDENCE_REVIEW_NOT_EXPERIMENT",
        "reference_shape": "16x8x2",
        "assumed_widths": "parameter=8,input=1,index=2 bytes",
        "provenance": {
            "crossover": "integrations/continual_learning/cl3t_crossover_analysis.py",
            "sensitivity": "integrations/continual_learning/cl3u_sensitivity.py",
            "components": "integrations/continual_learning/cl3v_cost_components.py",
            "visualization": "integrations/continual_learning/cl3r_visualization.py",
        },
        "claims": [
            {
                "id": "H1-S1",
                "status": "SUPPORTED_WITHIN_STATIC_SOFTWARE_PROXY_ONLY",
                "statement": "Sparse-local assumed inference payload is lower than dense at 2/16 active inputs.",
                "dense_bytes": dense2,
                "sparse_bytes": sparse2,
                "reduction_fraction": (dense2 - sparse2) / dense2,
            },
            {
                "id": "H1-S2",
                "status": "SUPPORTED_WITHIN_STATIC_SOFTWARE_PROXY_ONLY",
                "statement": "At 12/16 activity the modeled advantage is smaller; no strict advantage remains from 14/16.",
                "dense_bytes": dense12,
                "sparse_bytes": sparse12,
                "last_strictly_cheaper_activity": 13,
                "first_not_cheaper_activity": 14,
            },
            {
                "id": "H1-S3",
                "status": "MODEL_SENSITIVITY_NOT_HARDWARE_VALIDATION",
                "statement": "Nine counterfactual scenarios vary shape and packed widths; index visits can offset sparse savings.",
                "scenario_count": len(shape_sweep["scenarios"]),
                "nonreference_shapes_are_extrapolations": True,
            },
        ],
        "unsupported_claims": {
            "H1_physical_energy_reduction": "NOT_MEASURED",
            "H1_actual_DRAM_or_cache_traffic": "NOT_MEASURED",
            "H1_runtime_or_real_operation_count": "NOT_MEASURED",
            "H2_catastrophic_forgetting_reduction": "NOT_TESTED",
            "H3_accuracy_matched_energy_improvement": "NOT_TESTED",
            "photonic_or_Loihi_superiority": "NOT_TESTED",
        },
        "decision": "NO_GO_FOR_REAL_TRAINING_WITHOUT_SEPARATE_APPROVAL",
        "training_performed": False,
        "inference_performed": False,
        "energy_measured": False,
    }
