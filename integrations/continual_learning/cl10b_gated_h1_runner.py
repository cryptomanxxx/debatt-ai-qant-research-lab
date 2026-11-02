"""CL-10B: bounded H1 runner candidate. NO EXECUTION AUTHORIZED.

This module is import-safe and exposes no CLI path to model training. The
scientific core is prepared for separate independent review; the top-level
entrypoint rejects all invocations until a later, separately authorized change.
"""
from __future__ import annotations

import json
from pathlib import Path

BASE = Path(__file__).resolve().parent
PROPOSAL = BASE / "cl10a_first_h1_training_proposal.json"
ARMS = (
    "dense_sgd", "local_hybrid", "sparse_local_hybrid", "sparse_shared_index"
)
SEEDS = (101, 102, 103, 104, 105)
TASKS = ("T1", "T2", "T3", "T4")
CHECKPOINTS = ("pretrain", "post_T1", "post_T2", "post_T3", "post_T4")
MAX_UPDATES = 512


class ExecutionNotAuthorized(RuntimeError):
    pass


def validate_review_only_proposal(proposal: dict) -> dict:
    """Check the scope without constructing a dataset or importing models."""
    if type(proposal) is not dict or proposal.get("kind") != "CL10A_FIRST_BOUNDED_H1_TRAINING_PROPOSAL":
        raise ValueError("CL-10B: unexpected proposal")
    if (proposal.get("decision") != "NO_GO" or
            proposal.get("review_only") is not True or
            any(proposal.get(field) is not False for field in (
                "training_authorized", "inference_authorized",
                "dataset_generation_authorized", "runnable_workflow_created")) or
            proposal.get("observations_recorded") != 0):
        raise ValueError("CL-10B: proposal safety state changed")
    plan = proposal.get("training_plan", {})
    data = proposal.get("dataset", {})
    evaluation = proposal.get("evaluation", {})
    if (plan.get("arms") != list(ARMS) or plan.get("seeds") != list(SEEDS)
            or plan.get("tasks_order") != list(TASKS)
            or plan.get("updates_per_arm_seed_max") != MAX_UPDATES
            or plan.get("total_updates_max") != 10240
            or plan.get("automatic_retries") is not False
            or data.get("tasks") != list(TASKS)
            or data.get("train_per_task") != 128
            or data.get("heldout_per_task") != 64
            or evaluation.get("checkpoints") != list(CHECKPOINTS)
            or evaluation.get("heldout_tasks_each_checkpoint") != list(TASKS)):
        raise ValueError("CL-10B: proposed H1 workload changed")
    return {"decision": "NO_GO", "scope": "STATIC_REVIEW_ONLY",
            "arms": 4, "seeds": 5, "maximum_updates": 10240,
            "models_constructed": 0, "datasets_generated": 0}


def _future_seed_arm_core(seed: int, arm: str) -> dict:
    """UNAPPROVED scientific core. Do not call before external signoff.

    Outputs are in-memory, not an immutable experiment artifact. This function
    is deliberately unreachable through the CL-10B CLI/CI; a later reviewed
    runner and writer must govern it and freeze all data/numeric gates.
    """
    if type(seed) is not int or seed not in SEEDS or arm not in ARMS:
        raise ValueError("unplanned seed or arm")
    from integrations.continual_learning.cl2_mlp_baseline import TinyDenseMLP
    from integrations.continual_learning.cl2_local_mlp import LocalHebbianMLP
    from integrations.continual_learning.cl2_sparse_forward import SparseForwardLocalMLP
    from integrations.continual_learning.cl2_shared_index_forward import SharedIndexSparseForwardLocalMLP
    from integrations.continual_learning.cl2_shared_rule_candidate import (
        generate_shared_rule_candidate, shared_rule_dataset_sha256,
    )

    classes = {
        "dense_sgd": TinyDenseMLP,
        "local_hybrid": LocalHebbianMLP,
        "sparse_local_hybrid": SparseForwardLocalMLP,
        "sparse_shared_index": SharedIndexSparseForwardLocalMLP,
    }
    splits = generate_shared_rule_candidate(seed, train_per_task=128, test_per_task=64)
    if tuple(item.task_id for item in splits) != TASKS:
        raise ValueError("unexpected task order")
    model = classes[arm](seed=seed, learning_rate=0.05)
    if model.parameter_count != 154:
        raise ValueError("model size drift")
    checkpoints = []
    training = []
    updates = 0

    def evaluate(checkpoint: str) -> None:
        for task in splits:
            correct = sum(
                model.predict(ex.features) == ex.label for ex in task.test
            )
            checkpoints.append({
                "arm": arm, "seed": seed, "checkpoint": checkpoint,
                "task": task.task_id, "correct": correct,
                "eligible": len(task.test),
            })

    evaluate("pretrain")
    for task in splits:
        losses = []
        for example in task.train:
            if updates >= MAX_UPDATES:
                raise RuntimeError("planned update ceiling exceeded")
            losses.append(model.step(example.features, example.label))
            updates += 1
        training.append({
            "arm": arm, "seed": seed, "task": task.task_id,
            "updates": len(losses), "mean_loss": sum(losses) / len(losses),
        })
        evaluate("post_" + task.task_id)
    if updates != MAX_UPDATES or len(checkpoints) != 20:
        raise RuntimeError("incomplete planned workload")
    return {
        "arm": arm, "seed": seed,
        "dataset_sha256": shared_rule_dataset_sha256(splits),
        "updates": updates, "training": training,
        "heldout": checkpoints,
        "observed_h1_source_event_vector": None,
        "physical_energy": "NOT_MEASURED",
        "status": "UNAPPROVED_CORE_NO_IMMUTABLE_ARTIFACT",
    }


def main() -> None:
    # The first operation is intentionally a permanent execution guard.
    # Read the review-only JSON but NEVER import/construct models here.
    proposal = json.loads(PROPOSAL.read_text(encoding="utf-8"))
    print(json.dumps(validate_review_only_proposal(proposal), sort_keys=True))
    raise ExecutionNotAuthorized(
        "CL-10B NO_GO: scientific, network, artifact and separate Marcus "
        "execution authorization gates are open; no training/inference."
    )


if __name__ == "__main__":
    main()
