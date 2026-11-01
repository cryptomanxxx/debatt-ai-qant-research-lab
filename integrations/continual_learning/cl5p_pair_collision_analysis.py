"""CL-5P: exact *pairwise* collision probabilities; no data generation.

Analytic independence applies only to two fresh, independent samples with
balanced XOR labels and a fixed Bernoulli p. Observed duplicate rejections
come from sequential sampling with global uniqueness and changing class
quotas across tasks; they are not estimable from pair collision probability
alone. This module runs no generator, inference, training, or energy model.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from integrations.continual_learning.cl5n_empirical_data_diagnostics import analyze

SOURCE_RUN = 41311514598
BITS = 16


def pair_collision(p: float) -> dict:
    """Exactly calculate P(X == Y) for two IID draws in each distribution.

    For XOR(0,1), label=0 first-pair conditional masses are proportional to
    [(1-p)^2, p^2]; label=1 masses are [1/2, 1/2]. Remaining 14 bits IID.
    Under balanced labels, pair collisions arise only with matching classes.
    """
    if type(p) not in (float, int) or not math.isfinite(p) or not 0 < p < 1:
        raise ValueError("p must be a finite probability in (0,1)")
    q = 1 - p
    same_bit = p * p + q * q
    remainder = same_bit ** (BITS - 2)
    class0 = remainder * ((q ** 4 + p ** 4) / (q * q + p * p) ** 2)
    class1 = remainder / 2
    balanced = (class0 + class1) / 4
    raw = same_bit ** BITS
    return {
        "raw_bernoulli": raw,
        "class0": class0,
        "class1": class1,
        "balanced_xor": balanced,
        "effective_support": 1 / balanced,
    }


def analyze_existing(report: dict) -> list[dict]:
    """Read validated CL-5M aggregate only; pair prediction is not a fit."""
    rows = analyze(report)  # Fail-closed provenance and counter conservation.
    output = []
    for row in rows:
        prediction = pair_collision(row["nominal"])
        eligible = row["attempts"] - row["class_rejected"]
        if eligible <= 0:
            raise ValueError("no eligible candidate proposals")
        if row["accepted"] + row["duplicates"] != eligible:
            raise ValueError("counter conservation after class-full filtering failed")
        output.append({
            **row, **prediction,
            "eligible_proposals": eligible,
            "observed_duplicate_fraction": row["duplicates"] / eligible,
        })
    return output


def render_markdown(rows: list[dict]) -> str:
    lines = [
        "## CL-5P — exact pairwise collision and observed duplicates",
        "",
        "> **READ-ONLY ANALYSIS / NO DATA GENERATION / NO TRAINING / NO ENERGY MEASUREMENT.** "
        "Scientific status remains **NO_GO**.",
        "",
        f"Existing aggregate source: [CL-5M run {SOURCE_RUN}]"
        f"(https://github.com/cryptomanxxx/debatt-ai-qant-research-lab/actions/runs/{SOURCE_RUN}), seed 101.",
        "",
        "| Task | p | Raw IID pair collision | Balanced-XOR IID pair collision | Effective pairwise support | Duplicate rejects / eligible attempts |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for item in rows:
        lines.append(
            f"| {item['task']} | {item['nominal']:.1%} | "
            f"{item['raw_bernoulli']:.4%} | {item['balanced_xor']:.4%} | "
            f"{item['effective_support']:.0f} | "
            f"{item['duplicates']}/{item['eligible_proposals']} "
            f"({item['observed_duplicate_fraction']:.1%}) |"
        )
    lines.extend([
        "",
        "### Collision concentration (descriptive log-scaled bars)",
        "",
        "Bars show the approximate order of magnitude of the **two-draw** "
        "balanced-XOR collision probability (one block per half-decade), "
        "not duplicate-rejection rates.",
        "",
        "| Task | Collision concentration |",
        "|---|---|",
    ])
    for item in rows:
        # Longer means a larger pairwise collision probability.
        bar_length = max(1, round((5 + math.log10(item["balanced_xor"])) * 2))
        lines.append(f"| {item['task']} | {'█' * bar_length} |")
    lines.extend([
        "",
        "**Exact:** Probability of a collision between *two independent draws* "
        "from the specified Bernoulli or ideal equal-label XOR distribution.",
        "",
        "**Observed:** Duplicate rejects divided by candidates surviving the "
        "class-full check. The generator checks class quotas before uniqueness "
        "and maintains one global set across tasks and partitions.",
        "",
        "**Not inferred:** A pairwise collision probability is **not** the "
        "expected rejection fraction for an evolving finite sample. "
        "The aggregate counters lack the individual accepted patterns "
        "needed to compute exact conditional next-proposal rejection rates. "
        "No causal uniqueness-bias estimate, model quality, or physical energy "
        "claim is made.",
        "",
    ])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--source-run", required=True, type=int)
    parser.add_argument("--summary", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.source_run != SOURCE_RUN:
        parser.error("CL-5P requires the preregistered seed-101 source run")
    report = json.loads(args.report.read_text(encoding="utf-8"))
    summary = render_markdown(analyze_existing(report))
    with args.summary.open("a", encoding="utf-8") as handle:
        handle.write(summary)
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
