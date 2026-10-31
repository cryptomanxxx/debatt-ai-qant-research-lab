"""CL-3Q joint static source-cost review, NOT an experiment or energy study.

Combines two separately defined no-model-execution proxies: training-input
CL-3P and heldout-feature CL-3O. Never combines their bytes into joules.
"""
from __future__ import annotations

import argparse
import json

from integrations.continual_learning.cl3o_activity_weighted_inference import (
    prepare_activity_weighted_preview,
)
from integrations.continual_learning.cl3p_activity_weighted_training import (
    ARMS, prepare_training_proxy,
)


def compare_static_proxies() -> dict:
    """Per-seed, per-arm training vs inference summaries with separate scopes."""
    train = prepare_training_proxy()
    infer = prepare_activity_weighted_preview()
    if len(train["paired_seeds"]) != len(infer["paired_seeds"]):
        raise ValueError("seed schedules differ")
    summaries = []
    for tr_seed, in_seed in zip(train["paired_seeds"], infer["paired_seeds"]):
        if tr_seed["seed"] != in_seed["seed"]:
            raise ValueError("paired seed order mismatch")
        if len(tr_seed["tasks"]) != len(in_seed["tasks"]):
            raise ValueError("task schedules differ")
        counts = {arm: {
            "training_logical_payload_bytes": 0,
            "training_logical_parameter_reads": 0,
            "training_logical_parameter_writes": 0,
            "heldout_inference_logical_payload_bytes": 0,
            "heldout_inference_logical_parameter_reads": 0,
            "heldout_inference_logical_parameter_writes": 0,
        } for arm in ARMS}
        for a, b in zip(tr_seed["tasks"], in_seed["tasks"]):
            if a["task_id"] != b["task_id"]:
                raise ValueError("task order mismatch")
            training = a["training_input_proxy"]["by_arm"]
            inference = b["heldout_input_preview"]["by_arm"]
            for arm in ARMS:
                row = counts[arm]
                row["training_logical_payload_bytes"] += training[arm]["logical_payload_bytes"]
                row["training_logical_parameter_reads"] += training[arm]["parameter_reads"]
                row["training_logical_parameter_writes"] += training[arm]["parameter_writes"]
                row["heldout_inference_logical_payload_bytes"] += inference[arm]["logical_payload_bytes"]
                row["heldout_inference_logical_parameter_reads"] += inference[arm]["parameter_reads"]
                row["heldout_inference_logical_parameter_writes"] += inference[arm]["parameter_writes"]
        dense_train = counts["dense_sgd"]["training_logical_payload_bytes"]
        dense_infer = counts["dense_sgd"]["heldout_inference_logical_payload_bytes"]
        if dense_train <= 0 or dense_infer <= 0:
            raise ValueError("nonpositive dense baseline cost")
        for row in counts.values():
            row["training_payload_ratio_to_dense"] = row["training_logical_payload_bytes"] / dense_train
            row["heldout_inference_payload_ratio_to_dense"] = (
                row["heldout_inference_logical_payload_bytes"] / dense_infer
            )
        summaries.append({
            "seed": tr_seed["seed"],
            "hypothetical_training_updates_per_arm": tr_seed["hypothetical_updates_per_arm"],
            "heldout_feature_count_per_arm": sum(
                task["heldout_input_preview"]["input_examples"] for task in in_seed["tasks"]
            ),
            "by_arm": counts,
        })
    return {
        "kind": "CL3Q_SEPARATE_STATIC_SOURCE_COST_SCOPES_ONLY",
        "paired_seeds": summaries,
        "model_training_or_inference_performed": False,
        "accuracy_measured": False,
        "physical_energy_measured": False,
        "acceptable_accuracy_tradeoff_verified": False,
        "cost_ratios_are_energy_ratios": False,
        "note": "Training and heldout inference payloads are separately scoped logical-access proxies; no combined energy or model-quality claim is supported.",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CL-3Q no-model static cost review")
    parser.add_argument("--preview", action="store_true", required=True)
    args = parser.parse_args(argv)
    if args.preview:
        print(json.dumps(compare_static_proxies(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
