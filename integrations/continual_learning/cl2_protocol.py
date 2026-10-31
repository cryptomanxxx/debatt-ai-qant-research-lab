"""CL-2 draft protocol structural checks and offline metric definitions.

No dataset generation, model execution, imports from vendor SDKs, network
access or file writes. A validated draft is NOT compute authorization.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence


PROTOCOL_PATH = Path(__file__).with_name("cl2_protocol_draft.json")


def validate_draft(spec: Mapping[str, object]) -> None:
    """Reject contradictory or unsafe draft plans. Does not approve a plan."""
    if spec.get("status") != "draft_unapproved_unexecuted":
        raise ValueError("CL-2 remains an unapproved draft")
    if spec.get("execution_authorized") is not False:
        raise ValueError("draft cannot authorize compute")
    if spec.get("task_scenario") != "shared_binary_labels_no_task_id":
        raise ValueError("task identity or label scenario changed")
    order = spec.get("task_order")
    if not isinstance(order, list) or order != ["T1", "T2", "T3", "T4"]:
        raise ValueError("task order must be fixed T1 through T4")
    if spec.get("checkpoints") != ["before"] + [f"after_{t}" for t in order]:
        raise ValueError("all five checkpoints required")
    if spec.get("input_size") != 16 or spec.get("n_classes") != 2:
        raise ValueError("unexpected model/data dimensions")
    tasks = spec.get("tasks")
    if not isinstance(tasks, list) or len(tasks) != len(order):
        raise ValueError("four task definitions required")
    for task_id, task in zip(order, tasks):
        if not isinstance(task, dict) or task.get("id") != task_id:
            raise ValueError("task definitions must follow order")
        rate = task.get("activity_probability")
        if isinstance(rate, bool) or not isinstance(rate, (int, float)) or not 0 < rate <= 1:
            raise ValueError("invalid activity probability")
        if not isinstance(task.get("label_rule"), str) or not task["label_rule"]:
            raise ValueError("label rule must be documented, even if still draft")
    seeds = spec.get("seeds")
    if not isinstance(seeds, list) or len(seeds) < 3 or len(set(map(repr, seeds))) != len(seeds):
        raise ValueError("distinct paired seeds required")
    if any(type(s) is not int or s < 0 for s in seeds):
        raise ValueError("seeds must be nonnegative integers")
    data = spec.get("data")
    if not isinstance(data, dict) or data.get("test_labels_used_for_updates") is not False:
        raise ValueError("held-out labels must never update model weights")
    if any(type(data.get(k)) is not int or data[k] <= 0 for k in ("train_per_task", "test_per_task")):
        raise ValueError("positive fixed split sizes required")
    if data.get("split_policy") != "disjoint_seeded_precommitted":
        raise ValueError("split policy not fixed")
    generator = spec.get("generator")
    if not isinstance(generator, dict) or generator.get("download_allowed") is not False:
        raise ValueError("external dataset download forbidden")
    if generator.get("name") != "unimplemented_binary_generator" or generator.get("source_sha256") is not None:
        raise ValueError("draft generator must remain explicitly unimplemented")
    budget = spec.get("training_budget")
    if not isinstance(budget, dict) or any(
        budget.get(k) != 0 for k in ("max_external_requests", "max_hardware_calls")
    ):
        raise ValueError("no external requests or hardware calls allowed")
    expected_updates = len(order) * data["train_per_task"]
    if budget.get("max_updates_per_seed") != expected_updates:
        raise ValueError("update cap must agree with fixed training examples")
    if spec.get("energy_measurement") != "prohibited_until_calibrated_model":
        raise ValueError("physical energy claims forbidden")


def load_and_validate(path: Path = PROTOCOL_PATH) -> dict:
    """Parse a local proposal without invoking training or mutating state."""
    spec = json.loads(path.read_text(encoding="utf-8"))
    validate_draft(spec)
    return spec


def forgetting_by_task(
    accuracy: Sequence[Sequence[float]],
    task_order: Sequence[str] = ("T1", "T2", "T3", "T4"),
) -> dict[str, float]:
    """Calculate final forgetting from a five-checkpoint/four-task matrix.

    Row 0 is pretraining and rows 1..4 follow respective tasks; columns are
    fixed task order. No clamping: negative forgetting records improvement.
    Scores must lie in [0, 1]. Final-task forgetting is undefined, omitted.
    """
    n = len(task_order)
    if n < 2 or len(set(task_order)) != n or len(accuracy) != n + 1:
        raise ValueError("expected n+1 checkpoints and unique task IDs")
    for row in accuracy:
        if len(row) != n or any(
            isinstance(value, bool)
            or not isinstance(value, (float, int))
            or not 0 <= value <= 1
            for value in row
        ):
            raise ValueError("complete finite accuracy matrix in [0, 1] required")
    return {
        task_order[i]: max(accuracy[k][i] for k in range(i + 1, n)) - accuracy[n][i]
        for i in range(n - 1)
    }


if __name__ == "__main__":
    validated = load_and_validate()
    print(f"{validated['protocol_id']}: draft structurally valid; NOT approved or executed")
