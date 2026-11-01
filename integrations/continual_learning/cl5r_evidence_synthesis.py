"""CL-5R reproducible evidence synthesis from the existing CL-5M JSON report.

This read-only scientific analysis does not import generators or models and
never executes synthetic data creation, training, inference, or power probes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from integrations.continual_learning.cl5n_empirical_data_diagnostics import analyze
from integrations.continual_learning.cl5o_class_conditioning_analysis import compare_existing_report
from integrations.continual_learning.cl5p_pair_collision_analysis import analyze_existing
from integrations.continual_learning.cl5q_iid_occupancy_reference import analyze_report

SOURCE_RUN = 41311514598


def synthesize(report: dict) -> str:
    """Validate source contracts and classify evidence without causal overclaims."""
    actual = analyze(report)
    class_model = compare_existing_report(report)
    pair_model = analyze_existing(report)
    occupancy_model = analyze_report(report)
    if any([actual[i]["task"], class_model[i]["task"], pair_model[i]["task"],
            occupancy_model[i]["task"]].count(actual[i]["task"]) != 4
           for i in range(4)):
        raise ValueError("task correspondence failure")
    lines = [
        "# CL-5R: Scientific evidence synthesis — synthetic data selection",
        "",
        "> **Scientific status: NO_GO.** This is a post-hoc synthesis of one "
        "previously authorized seed-101 synthetic data-only run plus deterministic "
        "mathematical models. NO new data, training, inference, power measurement "
        "or scientific gate approval.",
        "",
        "## Primary evidence and reproducibility",
        "",
        f"- Source: [CL-5M workflow run {SOURCE_RUN}]"
        f"(https://github.com/cryptomanxxx/debatt-ai-qant-research-lab/actions/runs/{SOURCE_RUN})",
        f"- Seed: {report['seed']}; aggregate-only records: {report['records_checked']}.",
        f"- Dataset digest (original and instrumented): `{report['original_dataset_sha256']}`.",
        "- Two generator implementations produced matching record digests on this seed.",
        "- Original source report is downloaded rather than regenerating inputs.",
        "- Data records themselves are not included in this synthesis.",
        "",
        "## Distinct levels of evidence",
        "",
        "| Stage | What we have established | What we have NOT established |",
        "|---|---|---|",
        "| CL-5M | Actual, single-seed synthetic input activity; class balance; "
        "generator candidate/rejection counters | Any model learning or physical energy efficiency |",
        "| CL-5N | Reproducible aggregate measurements and read-only plots | "
        "Population statistics or a causal rejection model |",
        "| CL-5O | Exact balanced-XOR expected activity WITHOUT uniqueness rejection | "
        "Uniqueness-only effect behind observed residuals |",
        "| CL-5P | Exact collision probability for TWO IID draws from idealized distributions | "
        "Sequential rejection-until-unique rates |",
        "| CL-5Q | Exact expected distinct count and next-draw collision after n IID "
        "WITH-REPLACEMENT proposals | Actual class-quota/global-uniqueness generator dynamics |",
        "",
        "## Consolidated results",
        "",
        "| Task | Nominal activity | Balanced-XOR expectation | Observed activity | "
        "Unexplained residual (pp) | IID pair collision | Observed duplicate rejects | "
        "IID distinct after 192 proposals |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for i in range(4):
        a, b, c, d = actual[i], class_model[i], pair_model[i], occupancy_model[i]
        n192 = next(v for v in d["curve"] if v["n"] == 192)
        lines.append(
            f"| {a['task']} | {a['nominal']:.2%} | {b['class_balanced']:.2%} | "
            f"{a['observed']:.2%} | {100*b['unexplained_residual']:+.2f} | "
            f"{c['balanced_xor']:.4%} | {c['duplicates']} | "
            f"{n192['expected_distinct']:.1f} |"
        )
    attempts = sum(row["attempts"] for row in actual)
    accepted = sum(row["accepted"] for row in actual)
    rejected_class = sum(row["class_rejected"] for row in actual)
    rejected_duplicate = sum(row["duplicates"] for row in actual)
    if attempts != accepted + rejected_class + rejected_duplicate:
        raise ValueError("total generator counters do not conserve")
    lines += [
        "",
        f"Observed generator totals: **{attempts} proposals**, **{accepted} accepted**, "
        f"**{rejected_class} class-full rejects**, **{rejected_duplicate} duplicate rejects**.",
        "",
        "## How to interpret these values",
        "",
        "- The 192-proposal IID model is a distinct mathematical experiment from "
        "the observed **192 accepted records per task**. Do not equate them.",
        "- Each original proposal is Bernoulli-distributed. In actual generation, "
        "class-full checks come first, then duplicate checks against a single "
        "global used-vector set that persists across tasks and splits.",
        "- Idealized class-balanced IID models do not account for order-dependent "
        "quota depletion or uniqueness rejection. They are analytic reference "
        "distributions, not fitted predictions of the observed counter series.",
        "- The CL-5O residual cannot be causally attributed to uniqueness alone; "
        "sample variation and task-order effects also remain.",
        "- Extremely sparse input proposals concentrate probability on fewer "
        "likely patterns. This observation alone cannot predict sparse model "
        "learning quality, catastrophic forgetting, runtime or energy.",
        "",
        "## The Compute Paradox: hypothesis status",
        "",
        "| Hypothesis | Status | Missing measurement |",
        "|---|---|---|",
        "| H1: sparse computation reduces operational cost | **NO_GO / untested** | "
        "Paired model workload and measured logical operation counts, traffic, "
        "wall-power or comparable hardware proxies under equal accuracy |",
        "| H2: local learning mitigates catastrophic forgetting | **NO_GO / untested** | "
        "Approved continuous-learning train/test runs, retention and uncertainty |",
        "| H3: combined sparse, local and memory-near/photonic advantage | **NO_GO / untested** | "
        "Fair ablations with learning quality, conversion/memory/control overhead and "
        "hardware-supported energy characterization |",
        "",
        "## Recommended scientific next steps (not executed)",
        "",
        "1. Review exact sampling- and class-quota-selection mathematics or "
        "pre-register a strictly bounded multi-seed data-only follow-up, with "
        "separate user authorization if further generation is needed.",
        "2. Pre-register model-comparison endpoints, failure thresholds, uncertainty "
        "intervals and hardware measurement methodology.",
        "3. Obtain **separate explicit user approval** before any model training, "
        "inference on new data or physical energy experiments.",
        "",
        "**No model training has been authorized or performed by CL-5R.**",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--source-run", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args(argv)
    if args.source_run != SOURCE_RUN:
        parser.error("CL-5R is pinned to existing CL-5M run 41311514598")
    report = json.loads(args.report.read_text(encoding="utf-8"))
    rendered = synthesize(report)
    args.output.write_text(rendered, encoding="utf-8")
    if args.summary is not None:
        with args.summary.open("a", encoding="utf-8") as out:
            out.write(rendered)
    print("CL-5R evidence synthesis completed; scientific status NO_GO; no data/training")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
