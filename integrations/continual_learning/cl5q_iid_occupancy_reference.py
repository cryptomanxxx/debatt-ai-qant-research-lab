"""CL-5Q: exact finite-proposal IID occupancy reference for balanced XOR.

This reads an already generated CL-5M aggregate report only. It never
generates input vectors, samples random numbers, trains/infer models or
estimates energy.

For n independent WITH-REPLACEMENT proposals of the same task distribution,
this module computes E[distinct among n] and P(next proposal repeats one
of those n). It does NOT model rejection-until-unique, class quotas filling
over time, or the actual globally shared used-set across changing tasks.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from integrations.continual_learning.cl5p_pair_collision_analysis import (
    SOURCE_RUN, analyze_existing, pair_collision,
)

TAIL_BITS = 14
PROPOSAL_COUNTS = (0, 1, 16, 32, 64, 128, 192, 256)


def grouped_masses(p: float) -> tuple[tuple[int, float], ...]:
    """Exact probabilities for all 2^16 vectors, grouped by Hamming weight.

    One pair 00/11 is conditioned on XOR=0; two pair patterns 01/10
    each have weight 1/4 under XOR=1, since the labels have mass 1/2.
    """
    pair_collision(p)  # Validate probability and document shared baseline.
    q = 1 - p
    same = p * p + q * q
    groups = []
    for k in range(TAIL_BITS + 1):
        multiplicity = math.comb(TAIL_BITS, k)
        tail = p ** k * q ** (TAIL_BITS - k)
        groups.extend((
            (multiplicity, 0.5 * q * q / same * tail),
            (multiplicity, 0.5 * p * p / same * tail),
            (2 * multiplicity, 0.25 * tail),
        ))
    if not math.isclose(sum(m * w for m, w in groups), 1, abs_tol=1e-12):
        raise ValueError("analytic vector probabilities do not sum to one")
    return tuple(groups)


def occupancy(p: float, n: int) -> dict:
    """Exact expectation for IID proposals; n is proposals, not accepted uniques."""
    if type(n) is not int or n < 0 or n > 100_000:
        raise ValueError("n must be a nonnegative bounded integer")
    groups = grouped_masses(p)
    expected_distinct = 0.0
    duplicate_next = 0.0
    for multiplicity, w in groups:
        # Stable probability the pattern appeared at least once in n proposals.
        seen = -math.expm1(n * math.log1p(-w))
        expected_distinct += multiplicity * seen
        duplicate_next += multiplicity * w * seen
    expected_repeats = max(0.0, n - expected_distinct)
    return {
        "n": n,
        "expected_distinct": expected_distinct,
        "expected_repeats": expected_repeats,
        "next_duplicate_probability": duplicate_next,
    }


def analyze_report(report: dict) -> list[dict]:
    base = analyze_existing(report)  # Strict CL-5M provenance checks.
    return [{
        "task": row["task"],
        "p": row["nominal"],
        "observed_duplicate_rejects": row["duplicates"],
        "observed_eligible_proposals": row["eligible_proposals"],
        "curve": [occupancy(row["nominal"], n) for n in PROPOSAL_COUNTS],
    } for row in base]


def render_markdown(rows: list[dict]) -> str:
    lines = [
        "## CL-5Q — exact IID occupancy reference",
        "",
        "> **MATHEMATICAL / READ-ONLY / NO GENERATION / NO TRAINING / NO ENERGY.** "
        "H1 remains **NO_GO**.",
        "",
        f"Source: [existing CL-5M seed-101 aggregate run {SOURCE_RUN}]"
        f"(https://github.com/cryptomanxxx/debatt-ai-qant-research-lab/actions/runs/{SOURCE_RUN}).",
        "",
        "### Probability next proposal is a duplicate, after n independent IID draws",
        "",
        "| Task | n=0 | n=32 | n=64 | n=128 | n=192 | n=256 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        by_n = {value["n"]: value for value in row["curve"]}
        lines.append("| " + row["task"] + " | " + " | ".join(
            f"{by_n[n]['next_duplicate_probability']:.1%}"
            for n in (0, 32, 64, 128, 192, 256)
        ) + " |")
    lines += [
        "",
        "### IID occupancy curve at 192 proposals",
        "",
        "| Task | Expected distinct | Expected repeats | Next-proposal collision | Visual |",
        "|---|---:|---:|---:|---|",
    ]
    for row in rows:
        value = next(x for x in row["curve"] if x["n"] == 192)
        bar = "█" * round(30 * value["next_duplicate_probability"])
        lines.append(
            f"| {row['task']} | {value['expected_distinct']:.1f} | "
            f"{value['expected_repeats']:.1f} | "
            f"{value['next_duplicate_probability']:.1%} | {bar or '·'} |"
        )
    lines += [
        "",
        "### Separate observed CL-5M generator counters (not equivalent)",
        "",
        "| Task | Observed duplicate rejects / proposals surviving class-full check |",
        "|---|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['task']} | {row['observed_duplicate_rejects']}/"
            f"{row['observed_eligible_proposals']} |"
        )
    lines += [
        "",
        "**Exact for its stated model:** These occupancy values are expectations "
        "for n fresh, independent, with-replacement proposals from one "
        "fixed *balanced-XOR mixture* distribution. They use algebraic "
        "enumeration of mass groups; no data records are synthesized.",
        "",
        "**Important limitation:** The real generator draws Bernoulli proposals, "
        "then rejects based on dynamically filling class quotas and a shared "
        "set of globally accepted unique vectors across tasks/splits. Thus "
        "n proposals cannot be replaced with n accepted records, and the IID "
        "expectations cannot be used as an unbiased prediction or causal "
        "explanation of observed rejection counts.",
        "",
        "No model accuracy, learning, latency or hardware energy was measured.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--source-run", type=int, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.source_run != SOURCE_RUN:
        parser.error("CL-5Q is pinned to the existing CL-5M source run")
    report = json.loads(args.report.read_text(encoding="utf-8"))
    summary = render_markdown(analyze_report(report))
    with args.summary.open("a", encoding="utf-8") as handle:
        handle.write(summary)
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
