"""CL-5L opt-in instrumented *candidate* generator; NOT an approved dataset run.

This is a separate implementation, not a replacement or patch of the
reviewed CL-2 generator. No workflow or test invokes this entrypoint.
Empirical equivalence is unverified pending separate data-generation approval.
"""
from __future__ import annotations

import hashlib
import random

from integrations.continual_learning.cl2_data_candidate import (
    ACTIVITY_RATES, INPUT_SIZE, Example, TaskSplit,
)
from integrations.continual_learning.cl2_shared_rule_candidate import (
    VERSION, global_label,
)
from integrations.continual_learning.cl5j_disconnected_counter_ledger import (
    ProposalCounterLedger,
)
from integrations.continual_learning.cl5k_counter_decision_adapter import (
    account_proposed_candidate,
)


def generate_shared_rule_candidate_with_counters(
    seed: int, *, train_per_task: int = 128, test_per_task: int = 64,
) -> tuple[tuple[TaskSplit, ...], tuple[dict, ...]]:
    """Return records and counter reports only if explicitly invoked.

    This function intentionally does not run during import or in CI tests.
    Do not invoke without explicit approval for a bounded data-only audit.
    """
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if any(type(n) is not int or n <= 0 or n % 2 for n in (train_per_task, test_per_task)):
        raise ValueError("train/test counts must be positive and even")
    if len(ACTIVITY_RATES) * (train_per_task + test_per_task) > 2 ** INPUT_SIZE:
        raise ValueError("input space too small for globally unique records")

    used: set[tuple[int, ...]] = set()
    tasks: list[TaskSplit] = []
    reports: list[dict] = []
    for task_id, probability in ACTIVITY_RATES.items():
        material = hashlib.sha256(f"{VERSION}|{seed}|{task_id}".encode("ascii")).digest()
        rng = random.Random(int.from_bytes(material, "big"))
        partitions: list[tuple[Example, ...]] = []
        for partition_name, count in (("train", train_per_task), ("test", test_per_task)):
            remaining = [count // 2, count // 2]
            records: list[Example] = []
            ledger = ProposalCounterLedger(attempt_cap=1_000_000)
            for _ in range(1_000_000):
                if not any(remaining):
                    break
                features = tuple(int(rng.random() < probability) for _ in range(INPUT_SIZE))
                label = global_label(features)
                decision = account_proposed_candidate(
                    ledger,
                    remaining_for_label=remaining[label],
                    duplicate_lookup_if_open=lambda: features in used,
                )
                if decision != "accepted":
                    continue
                used.add(features)
                records.append(Example(features, label))
                remaining[label] -= 1

            report = ledger.finish_partition(quotas_filled=not any(remaining))
            # An explicitly invoked connected generator has generated data.
            # Override the disconnected ledger's event-only provenance.
            report["kind"] = "CL5L_INSTRUMENTED_GENERATOR_PARTITION"
            report["generator_connected"] = True
            report["dataset_generation_executed"] = True
            report["task_id"] = task_id
            report["partition"] = partition_name
            reports.append(report)
            if any(remaining):
                raise ValueError("insufficient unique class-balanced records")
            partitions.append(tuple(records))
        tasks.append(TaskSplit(task_id, partitions[0], partitions[1]))
    return tuple(tasks), tuple(reports)
