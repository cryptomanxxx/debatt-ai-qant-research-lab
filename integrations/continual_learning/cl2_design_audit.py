"""Pure CL-2 candidate-design audit. No training, energy model or SDK usage."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product

from integrations.continual_learning.cl2_data_candidate import (
    INPUT_SIZE,
    TASK_PAIRS,
    TaskSplit,
)


@dataclass(frozen=True)
class SplitAudit:
    task_id: str
    partition: str
    examples: int
    positive_fraction: float
    observed_spike_fraction: float


def audit_candidate_splits(splits: tuple[TaskSplit, ...]) -> tuple[SplitAudit, ...]:
    """Check record integrity and measure observed (not configured) activity."""
    if tuple(s.task_id for s in splits) != tuple(TASK_PAIRS):
        raise ValueError("missing, reordered or repeated tasks")
    seen: set[tuple[int, ...]] = set()
    report: list[SplitAudit] = []
    for split in splits:
        pair = TASK_PAIRS[split.task_id]
        for name, rows in (("train", split.train), ("test", split.test)):
            if not rows:
                raise ValueError("empty partition")
            positives = 0
            spikes = 0
            for row in rows:
                x = row.features
                if (
                    len(x) != INPUT_SIZE
                    or any(type(v) is not int or v not in (0, 1) for v in x)
                    or type(row.label) is not int
                    or row.label != (x[pair[0]] ^ x[pair[1]])
                ):
                    raise ValueError("invalid features or task-specific label")
                if x in seen:
                    raise ValueError("duplicate feature vector across splits/tasks")
                seen.add(x)
                positives += row.label
                spikes += sum(x)
            report.append(
                SplitAudit(
                    task_id=split.task_id,
                    partition=name,
                    examples=len(rows),
                    positive_fraction=positives / len(rows),
                    observed_spike_fraction=spikes / (len(rows) * INPUT_SIZE),
                )
            )
    return tuple(report)


def exact_uniform_rule_conflict(task_a: str, task_b: str) -> float:
    """Fraction of all 16-bit vectors on which two task label rules disagree.

    Enumerates only relevant bit positions (at most four), giving the exact
    proportion under a *uniform* 16-bit reference distribution, not the
    class-balanced or Bernoulli distributions used by the proposed sampler.
    """
    if task_a not in TASK_PAIRS or task_b not in TASK_PAIRS:
        raise ValueError("unknown task ID")
    indices = tuple(sorted(set(TASK_PAIRS[task_a] + TASK_PAIRS[task_b])))
    disagreement = 0
    for values in product((0, 1), repeat=len(indices)):
        x = dict(zip(indices, values))
        pa, pb = TASK_PAIRS[task_a], TASK_PAIRS[task_b]
        disagreement += (x[pa[0]] ^ x[pa[1]]) != (x[pb[0]] ^ x[pb[1]])
    return disagreement / (2 ** len(indices))
