"""CL-5O: analytical class-conditioning baseline for existing CL-5M observations.

No generator imports, data generation, model training, inference, or energy model.
The conditioned baseline is an exact expectation for balanced XOR labels under
independent Bernoulli input proposals *without* uniqueness rejection.
Its residual is NOT an identified causal effect of uniqueness: finite-sample
variation and order-dependent cross-task rejection also remain.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from integrations.continual_learning.cl5n_empirical_data_diagnostics import analyze

SOURCE_RUN = 41311514598
INPUT_BITS = 16


def xor_balanced_activity(p: float) -> float:
    """E[fraction active | 50/50 XOR(0,1) labels], absent uniqueness rejection.

    The 14 unrelated bits contribute 14p. Conditional XOR=0, bit pair 00
    or 11 has expected active bits 2p^2/(p^2+(1-p)^2); XOR=1 contributes 1.
    """
    if type(p) not in (int, float) or not math.isfinite(p) or not 0 < p < 1:
        raise ValueError("p must be a finite probability strictly between 0 and 1")
    q = 1 - p
    return (14 * p + p * p / (p * p + q * q) + 0.5) / INPUT_BITS


def compare_existing_report(report: dict) -> list[dict]:
    """Reuse CL-5N fail-closed validation; produce three-level comparisons."""
    baseline = analyze(report)
    return [
        {
            **row,
            "class_balanced": xor_balanced_activity(row["nominal"]),
            "class_conditioning_shift":
                xor_balanced_activity(row["nominal"]) - row["nominal"],
            "unexplained_residual":
                row["observed"] - xor_balanced_activity(row["nominal"]),
        }
        for row in baseline
    ]


def render_markdown(rows: list[dict]) -> str:
    lines = [
        "## CL-5O — class-balance conditioning versus observed activity",
        "",
        "> **READ-ONLY POST-HOC ANALYSIS.** No synthetic data generation, "
        "inference, training, hardware energy measurement or scientific approval. "
        "H1 remains **NO_GO**.",
        "",
        f"Source: [CL-5M run {SOURCE_RUN}]"
        f"(https://github.com/cryptomanxxx/debatt-ai-qant-research-lab/actions/runs/{SOURCE_RUN}), "
        "seed 101; 16 input bits, XOR(bits 0,1), balanced labels.",
        "",
        "| Task | Raw Bernoulli | Balanced XOR expectation | Observed | Conditioning shift (pp) | Unexplained residual (pp) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['task']} | {row['nominal']:.2%} | "
            f"{row['class_balanced']:.2%} | {row['observed']:.2%} | "
            f"{100 * row['class_conditioning_shift']:+.2f} | "
            f"{100 * row['unexplained_residual']:+.2f} |"
        )
    lines.extend([
        "",
        "### Activity comparison (each block ≈ 2 percentage points)",
        "",
        "| Task | Raw proposal | Balanced XOR baseline | Observed records |",
        "|---|---|---|---|",
    ])
    for row in rows:
        bars = lambda value: "█" * round(50 * value)
        lines.append(
            f"| {row['task']} | {bars(row['nominal'])} | "
            f"{bars(row['class_balanced'])} | {bars(row['observed'])} |"
        )
    lines.extend([
        "",
        "**Interpretation:** The raw-vs-balanced difference is mathematically "
        "attributable to enforcing equal XOR class counts under ideal independent "
        "Bernoulli proposals, before uniqueness filtering.",
        "",
        "**Not identified:** The observed-minus-balanced residual combines effects "
        "of global uniqueness rejection, finite-sample randomness, and "
        "sequential task/split order. It must **not** be presented as a measured "
        "uniqueness-only causal effect.",
        "",
        "No physical memory access, neural operation, energy use, or model learning "
        "is measured. Multiple independent preregistered seeds or exact "
        "without-replacement modeling would be needed for further attribution.",
        "",
    ])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--source-run", type=int, required=True)
    args = parser.parse_args(argv)
    if args.source_run != SOURCE_RUN:
        parser.error("source run must be the pinned CL-5M seed-101 run")
    report = json.loads(args.report.read_text(encoding="utf-8"))
    rendered = render_markdown(compare_existing_report(report))
    with args.summary.open("a", encoding="utf-8") as handle:
        handle.write(rendered)
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
