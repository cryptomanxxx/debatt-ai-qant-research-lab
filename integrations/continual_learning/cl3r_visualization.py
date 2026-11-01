"""CL-3R offline static research visualizations. NO training or model inference.

Produces SVG, HTML, JSON and GitHub Actions Markdown from pure CL-2/CL-3
source-level proxies. No model instances, datasets, vendor SDKs or network.
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

from integrations.continual_learning.cl3e_measurement_ledger import project_training_only
from integrations.continual_learning.cl3n_inference_accounting import estimate_prediction
from integrations.continual_learning.cl3t_crossover_analysis import analyze_crossover
from integrations.continual_learning.cl3u_sensitivity import sensitivity, check_reference
from integrations.continual_learning.cl3v_cost_components import component_report

ARMS = ("dense_sgd", "local_hybrid", "sparse_local_hybrid", "sparse_shared_index")
NAMES = {
    "dense_sgd": "Dense SGD",
    "local_hybrid": "Local hybrid",
    "sparse_local_hybrid": "Sparse local",
    "sparse_shared_index": "Shared-index sparse",
}
COLORS = {
    "dense_sgd": "#4b8ef4",
    "local_hybrid": "#30bfa0",
    "sparse_local_hybrid": "#e4a43c",
    "sparse_shared_index": "#b27af3",
}
DISCLAIMER = (
    "STATIC SOFTWARE PROXY ONLY — NOT model accuracy, hardware memory traffic, "
    "training, physical energy, measured latency, photonic or Loihi performance."
)


def build_preview() -> dict:
    """Only pure arithmetic over assumed active bits 0..16; no experiment."""
    series = []
    for active in range(17):
        by_arm = {}
        for arm in ARMS:
            train = project_training_only(arm, step_count=1, active_inputs_per_step=active)
            infer = estimate_prediction(arm, active)
            by_arm[arm] = {
                "training_logical_payload_bytes_per_assumed_update": train.logical_payload_bytes,
                "training_logical_parameter_reads": train.logical_parameter_reads,
                "training_logical_parameter_writes": train.logical_parameter_writes,
                "inference_logical_payload_bytes_per_assumed_prediction": infer.logical_payload_bytes,
                "inference_logical_parameter_reads": infer.parameter_reads,
                "inference_logical_parameter_writes": infer.parameter_writes,
            }
        series.append({"active_input_bits": active, "by_arm": by_arm})
    return {
        "kind": "CL3R_STATIC_VISUALIZATION_NO_COMPUTE",
        "disclaimer": DISCLAIMER,
        "training_executed": False,
        "model_inference_executed": False,
        "physical_energy_measured": False,
        "width_assumptions": {"parameter_bytes": 8, "input_bytes": 1, "active_index_bytes": 2},
        "series": series,
    }


def _svg_begin(width: int, height: int) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img">',
        '<rect width="100%" height="100%" fill="#101928"/>',
        '<style>text{font:14px Arial,sans-serif;fill:#e7eefc}.muted{fill:#a7bad5}.axis{stroke:#60738f;stroke-width:1}</style>',
    ]


def line_svg(preview: dict, metric: str, title: str) -> str:
    """Static SVG line chart with axes, legend, labels and explicit units."""
    width, height = 900, 510
    left, top, right, bottom = 90, 105, 35, 75
    plotw, ploth = width - left - right, height - top - bottom
    maximum = max(point["by_arm"][arm][metric] for point in preview["series"] for arm in ARMS)
    ceiling = max(1, maximum * 1.08)
    output = _svg_begin(width, height)
    output.append(f'<text x="35" y="35" font-size="21">{html.escape(title)}</text>')
    output.append('<text x="35" y="59" class="muted">Source-level logical payload bytes — not joules or measured DRAM bytes</text>')
    for step in range(5):
        val = ceiling * step / 4
        y = top + ploth * (1 - step / 4)
        output.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#31415a"/>')
        output.append(f'<text x="12" y="{y+5:.1f}" class="muted">{val:.0f}</text>')
    for active in (0, 4, 8, 12, 16):
        x = left + plotw * active / 16
        output.append(f'<text x="{x-5:.1f}" y="{height-bottom+26}" class="muted">{active}</text>')
    output.append(f'<text x="{left+plotw/2-85:.1f}" y="{height-15}">Active input bits (of 16)</text>')
    for idx, arm in enumerate(ARMS):
        color = COLORS[arm]
        points = []
        for point in preview["series"]:
            x = left + plotw * point["active_input_bits"] / 16
            y = top + ploth * (1 - point["by_arm"][arm][metric] / ceiling)
            points.append(f"{x:.1f},{y:.1f}")
        output.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="3"/>')
        lx = 37 + (idx % 2) * 225
        ly = 80 + (idx // 2) * 20
        output.append(f'<line x1="{lx}" y1="{ly-4}" x2="{lx+26}" y2="{ly-4}" stroke="{color}" stroke-width="4"/>')
        output.append(f'<text x="{lx+34}" y="{ly}">{html.escape(NAMES[arm])}</text>')
    output.append('</svg>')
    return "\n".join(output)


def bar_svg(preview: dict, active: int = 2) -> str:
    """Two independent phase groups; never sums training and inference."""
    if type(active) is not int or not 0 <= active <= 16:
        raise ValueError("active bits outside [0,16]")
    point = preview["series"][active]["by_arm"]
    metric_keys = (
        "training_logical_payload_bytes_per_assumed_update",
        "inference_logical_payload_bytes_per_assumed_prediction",
    )
    max_value = max(point[a][k] for a in ARMS for k in metric_keys)
    out = _svg_begin(950, 490)
    out.append(f'<text x="35" y="35" font-size="21">Phase-separated logical payload (activity = {active}/16)</text>')
    out.append('<text x="35" y="60" class="muted">Training assumes one update; inference assumes one prediction. No model runs.</text>')
    for j, (metric, phase) in enumerate(zip(metric_keys, ("Training (assumed update)", "Inference (assumed prediction)"))):
        left = 42 + j * 465
        out.append(f'<text x="{left}" y="110" font-size="17">{phase}</text>')
        for i, arm in enumerate(ARMS):
            y = 145 + i * 70
            value = point[arm][metric]
            length = 295 * value / max_value
            out.append(f'<text x="{left}" y="{y}" class="muted">{html.escape(NAMES[arm])}</text>')
            out.append(f'<rect x="{left}" y="{y+10}" width="{length:.1f}" height="19" rx="4" fill="{COLORS[arm]}"/>')
            out.append(f'<text x="{left+length+8:.1f}" y="{y+25}">{value}</text>')
    out.append('<text x="35" y="465" class="muted">Unit: assumed packed logical access bytes (not physical transfer or energy)</text>')
    out.append('</svg>')
    return "\n".join(out)


def architecture_svg() -> str:
    out = _svg_begin(900, 520)
    out.append('<text x="30" y="38" font-size="22">CL research architecture — governance-first</text>')
    nodes = [
        (40, 90, 370, 76, "CL-2 candidate models", "Dense / local / sparse ablations"),
        (490, 90, 370, 76, "Synthetic data audits", "CL-3C / G / M: no training"),
        (40, 230, 370, 76, "Training source proxies", "CL-3P: hypothetical updates"),
        (490, 230, 370, 76, "Inference source proxies", "CL-3N / O: hypothetical predicts"),
        (265, 380, 370, 80, "CL-3Q / R: review & charts", "NO_GO — experimental training blocked"),
    ]
    for x, y, w, h, title, subtitle in nodes:
        out.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="#23354d" stroke="#54799d"/>')
        out.append(f'<text x="{x+17}" y="{y+31}" font-size="19">{html.escape(title)}</text>')
        out.append(f'<text x="{x+17}" y="{y+55}" class="muted">{html.escape(subtitle)}</text>')
    for path in (
        "M225 166 V230", "M675 166 V230", "M225 306 V345 H450 V380",
        "M675 306 V345 H450 V380",
    ):
        out.append(f'<path d="{path}" fill="none" stroke="#6bbfc1" stroke-width="3"/>')
    out.append('</svg>')
    return "\n".join(out)


def _bar(value: int, maximum: int, width: int = 18) -> str:
    """GitHub-summary-safe Unicode bar: no external image URL needed."""
    if type(value) is not int or type(maximum) is not int or maximum <= 0 or value < 0 or value > maximum:
        raise ValueError("invalid bar range")
    filled = round(width * value / maximum)
    return "█" * filled + "░" * (width - filled)


def _sparkline(values: list[int]) -> str:
    """Compact source-proxy activity curve for GitHub Markdown job summaries."""
    if not values or any(type(value) is not int or value < 0 for value in values):
        raise ValueError("invalid sparkline values")
    lo, hi = min(values), max(values)
    levels = "▁▂▃▄▅▆▇█"
    return "".join(levels[round(7 * (v - lo) / (hi - lo))] if hi != lo else levels[0] for v in values)


def report_files(preview: dict) -> dict[str, str]:
    """Build all report contents in memory for testability."""
    chosen = preview["series"][2]["by_arm"]
    summary = [
        "# CL-3R — Static research visualization",
        "",
        "> " + DISCLAIMER,
        "",
        "## Logical payload per assumed operation at 2 active inputs",
        "",
        "| Architecture | Training bytes / assumed update | Inference bytes / assumed prediction |",
        "|---|---:|---:|",
    ]
    for arm in ARMS:
        row = chosen[arm]
        summary.append(
            f'| {NAMES[arm]} | {row["training_logical_payload_bytes_per_assumed_update"]} | '
            f'{row["inference_logical_payload_bytes_per_assumed_prediction"]} |'
        )
    summary.extend([
        "",
        "**No model training, model inference, accuracy or physical energy measurement occurred.**",
        "Download the `cl3r-static-research-visuals` artifact below for HTML, SVG and JSON.",
    ])
    summary.extend([
        "",
        "## Inline graphs (visible directly in this GitHub Actions summary)",
        "",
        "**Bars: 2 active input bits of 16.** Longer bar = more assumed logical payload bytes.",
        "",
        "| Architecture | Training / assumed update | Inference / assumed prediction |",
        "|---|---|---|",
    ])
    train_max = max(chosen[arm]["training_logical_payload_bytes_per_assumed_update"] for arm in ARMS)
    infer_max = max(chosen[arm]["inference_logical_payload_bytes_per_assumed_prediction"] for arm in ARMS)
    for arm in ARMS:
        row = chosen[arm]
        train_bytes = row["training_logical_payload_bytes_per_assumed_update"]
        infer_bytes = row["inference_logical_payload_bytes_per_assumed_prediction"]
        summary.append(
            f'| {NAMES[arm]} | {_bar(train_bytes, train_max)} {train_bytes} | '
            f'{_bar(infer_bytes, infer_max)} {infer_bytes} |'
        )
    summary.extend([
        "",
        "## Activity sensitivity curves (0 → 16 active input bits)",
        "",
        "These are Unicode sparklines, not measured power or energy graphs.",
        "",
        "| Architecture | Training: logical bytes | Inference: logical bytes |",
        "|---|---|---|",
    ])
    for arm in ARMS:
        training_values = [
            row["by_arm"][arm]["training_logical_payload_bytes_per_assumed_update"]
            for row in preview["series"]
        ]
        inference_values = [
            row["by_arm"][arm]["inference_logical_payload_bytes_per_assumed_prediction"]
            for row in preview["series"]
        ]
        summary.append(
            f'| {NAMES[arm]} | {_sparkline(training_values)} | {_sparkline(inference_values)} |'
        )
    summary.extend([
        "",
        "The full-resolution SVG plots and architecture diagram remain in the downloadable artifact.",
    ])
    crossover = analyze_crossover(preview)
    summary.extend([
        "",
        "## When does the logical payload advantage disappear?",
        "",
        "Discrete source-level breakpoints relative to dense SGD (not hardware energy).",
        "",
        "| Phase | Architecture | Last better at | First not better at |",
        "|---|---|---:|---:|",
    ])
    for item in crossover["comparisons"]:
        last = item["last_strictly_better_activity"]
        first = item["first_not_better_activity"]
        summary.append(
            f'| {item["phase"]} | {NAMES[item["arm"]]} | '
            f'{last if last is not None else "Never"} | '
            f'{first if first is not None else "Not within 0–16"} |'
        )
    summary.extend([
        "",
        "Values are counts of active bits out of 16; equality is not a saving. "
        "No measured energy or learning quality is implied.",
    ])
    check_reference()
    sensitivity_result = sensitivity()
    summary.extend([
        "",
        "## CL-3U: counterfactual shape and width sensitivity (not measured energy)",
        "",
        "Other shapes extrapolate source-loop algebra; only 16x8x2 is cross-checked against CL-2/CL-3. The table shows the sparse-local inference cutoff in active inputs.",
        "",
        "| Input/hidden/output | Parameter/input/index bytes | Last cheaper | First not cheaper |",
        "|---|---|---:|---:|",
    ])
    for scenario in sensitivity_result["scenarios"]:
        dims = scenario["shape"]
        widths = scenario["widths"]
        comparison = next(row for row in scenario["comparisons"] if row["phase"] == "inference" and row["arm"] == "sparse_local_hybrid")
        summary.append(
            f'| {dims["inputs"]}/{dims["hidden"]}/{dims["outputs"]} | '
            f'{widths["parameter"]}/{widths["input"]}/{widths["index"]} | '
            f'{comparison["last_strict_advantage"] if comparison["last_strict_advantage"] is not None else "Never"} | '
            f'{comparison["first_not_better"] if comparison["first_not_better"] is not None else "Beyond sweep"} |'
        )
    summary.extend(["", "These are **counterfactual software-proxy** comparisons only: no network resizing, training, inference, memory traffic or hardware energy was measured."])
    components_result = component_report()
    summary.extend([
        "",
        "## CL-3V: where do logical payload bytes come from?",
        "",
        "Assumed **logical** parameter, input-check and active-index payload at two activity levels; "
        "P/I/X bars are proportional *within each row*, not measured energy.",
        "",
        "| Phase | Activity | Architecture | Parameters (P) | Inputs (I) | Indices (X) | Total bytes | Mix |",
        "|---|---:|---|---:|---:|---:|---:|---|",
    ])
    for phase in ("training", "inference"):
        for active in (2, 12):
            for arm in ARMS:
                row = next(x for x in components_result["rows"]
                           if x["phase"] == phase and x["active"] == active and x["arm"] == arm)
                total = row["total_bytes"]
                sizes = [row["parameter_bytes"], row["input_bytes"], row["index_bytes"]]
                # Allocate twelve glyphs with exact total, preserving a stable order.
                lengths = [12 * value // total for value in sizes]
                remain = 12 - sum(lengths)
                for idx in sorted(range(3), key=lambda i: (12 * sizes[i] % total, -i), reverse=True)[:remain]:
                    lengths[idx] += 1
                mix = "P" * lengths[0] + "I" * lengths[1] + "X" * lengths[2]
                summary.append(
                    f'| {phase} | {active}/16 | {NAMES[arm]} | '
                    f'{sizes[0]} | {sizes[1]} | {sizes[2]} | {total} | {mix} |'
                )
    summary.extend([
        "",
        "P = parameter payload, I = input visits, X = index visits. "
        "Zero-height categories may be invisible in a 12-character bar. "
        "The full 0–16 activity sweep is available in the artifact JSON. "
        "**No DRAM traffic or energy has been measured.**",
    ])
    arch = architecture_svg()
    train_line = line_svg(preview, "training_logical_payload_bytes_per_assumed_update", "Hypothetical training-step logical payload")
    infer_line = line_svg(preview, "inference_logical_payload_bytes_per_assumed_prediction", "Hypothetical inference logical payload")
    bars = bar_svg(preview)
    sections = [
        ("Research architecture", arch),
        ("Phase-separated cost comparison", bars),
        ("Training sensitivity", train_line),
        ("Inference sensitivity", infer_line),
    ]
    body = [
        '<!doctype html><html lang="en"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width,initial-scale=1">',
        '<title>CL-3R static research visualization</title>',
        '<style>body{background:#0c1320;color:#e7eefc;font:16px system-ui,sans-serif;margin:0 auto;max-width:1050px;padding:24px}section{margin:26px 0}svg{width:100%;height:auto;border-radius:10px}small{color:#acc0da}</style>',
        '</head><body><h1>CL-3R — Static research visualization</h1>',
        '<p><strong>NO TRAINING / NO REAL ENERGY MEASUREMENT</strong></p>',
        f'<p><small>{html.escape(DISCLAIMER)}</small></p>',
    ]
    for label, svg in sections:
        body.extend([f'<section><h2>{html.escape(label)}</h2>', svg, '</section>'])
    body.append('</body></html>')
    return {
        "summary.md": "\n".join(summary) + "\n",
        "report.html": "\n".join(body),
        "architecture.svg": arch,
        "comparison.svg": bars,
        "training-activity.svg": train_line,
        "inference-activity.svg": infer_line,
        "projections.json": json.dumps(preview, sort_keys=True, indent=2) + "\n",
        "sensitivity.json": json.dumps(sensitivity_result, sort_keys=True, indent=2) + "\n",
        "components.json": json.dumps(components_result, sort_keys=True, indent=2) + "\n",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Offline CL-3R static graphs only")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    result = report_files(build_preview())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for filename, contents in result.items():
        (args.output_dir / filename).write_text(contents, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
