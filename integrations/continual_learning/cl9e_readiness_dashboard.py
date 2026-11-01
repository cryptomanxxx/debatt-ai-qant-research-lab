"""CL-9E: read-only scientific readiness dashboard. Never imports model code."""
from __future__ import annotations

import argparse
import json
import html
from pathlib import Path

from integrations.continual_learning.cl7a_scientific_readiness_review import EXPECTED_FINDINGS

BASE = Path(__file__).resolve().parent

def _markdown_cell(value: str) -> str:
    """Render contract text as one inert Markdown table cell."""
    if not isinstance(value, str):
        raise ValueError("Invalid table-cell source")
    normalized = " ".join(value.split())
    return (html.escape(normalized, quote=True)
            .replace("|", "&#124;")
            .replace(chr(96), "&#96;")
            .replace("*", "&#42;")
            .replace("[", "&#91;")
            .replace("]", "&#93;"))


def _load(name: str) -> dict:
    value = json.loads((BASE / name).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Invalid review contract: {name}")
    return value


def snapshot() -> dict:
    readiness = _load("cl9a_minimal_h1_readiness.json")
    source = _load("cl7f_h1_decision_matrix.json")
    critique = _load("cl7a_scientific_design_critique.json")
    acceptance = _load("cl9b_h1_acceptance_contract.json")
    measurement = _load("cl9c_measurement_protocol.json")
    reporting = _load("cl9d_reproducibility_failure_policy.json")
    host = _load("cl8h_alternative_host_security_contract.json")
    scientific = _load("cl8a_independent_h1_review_request.json")
    contracts = (readiness, source, critique, acceptance, measurement,
                 reporting, host, scientific)
    if any(contract.get("decision") != "NO_GO" for contract in contracts):
        raise ValueError("Inconsistent scientific decision: refuse GO report")
    gates = source["readiness_gates"]
    findings = critique["findings"]
    if (len(gates) != 9 or len({g["id"] for g in gates}) != 9
            or any(g.get("status") != "OPEN" for g in gates)
            or [f.get("id") for f in findings] != list(EXPECTED_FINDINGS)
            or any(f.get("status") != "OPEN" for f in findings)
            or [g["id"] for g in readiness["minimum_gates"]] !=
            [g["id"] for g in gates]
            or any(g["status"] != "OPEN" for g in readiness["minimum_gates"])
            or acceptance.get("model_training_authorized") is not False
            or measurement.get("model_training_authorized") is not False
            or reporting.get("training_authorized") is not False
            or host.get("training_authorized") is not False
            or scientific.get("independent_scientific_signoff") is not False):
        raise ValueError("Readiness gates not established; refuse misleading summary")
    return {
        "decision": "NO_GO",
        "scientific_gates": gates,
        "blocking_findings": findings,
        "documents": [
            ("CL-9A", "Minimal H1 scientific readiness", "cl9a_minimal_h1_readiness.json"),
            ("CL-9B", "Acceptance and negative-result rules", "cl9b_h1_acceptance_contract.json"),
            ("CL-9C", "Measurement record protocol", "cl9c_measurement_protocol.json"),
            ("CL-9D", "Failure and reproducibility contract", "cl9d_reproducibility_failure_policy.json"),
            ("CL-8H", "Host acceptance requirements", "cl8h_alternative_host_security_contract.json"),
        ],
        "new_training_runs": 0,
        "actual_energy_measured": False,
        "independent_scientific_signoff": False,
    }


def summary_markdown(data: dict) -> str:
    base = ("https://github.com/cryptomanxxx/"
            "debatt-ai-qant-research-lab/blob/main/integrations/continual_learning/")
    lines = [
        "# CL-9E — Scientific readiness dashboard",
        "",
        "> **NO_GO — REVIEW ONLY.** No model execution, training, inference, new data,",
        "> hardware energy measurement or external scientific endorsement.",
        "",
        "## Scientific gates: 0 / 9 closed",
        "",
        "| Gate | Status | Next evidence |",
        "| --- | --- | --- |",
    ]
    for gate in data["scientific_gates"]:
        lines.append(f'| {_markdown_cell(gate["id"])} | 🔴 OPEN | {_markdown_cell(gate["next_evidence"])} |')
    lines += [
        "",
        "## Scientific design findings: 0 / 6 resolved",
        "",
        "| Finding | Status |",
        "| --- | --- |",
    ]
    for finding in data["blocking_findings"]:
        lines.append(f'| {_markdown_cell(finding["id"])} | 🔴 OPEN |')
    lines += [
        "",
        "## Prepared documentation (not evidence of gate closure)",
        "",
        "| Stage | Prepared contract |",
        "| --- | --- |",
    ]
    for stage, label, path in data["documents"]:
        lines.append(f"| {stage} | [{label}]({base}{path}) |")
    lines += [
        "",
        "## Next practical steps",
        "",
        "1. Freeze H1 source-event definitions, numeric parity tolerance, train-only controls and the data-selection audit.",
        "2. Obtain actual independent scientific critique and a verified bounded offline execution environment.",
        "3. Only then ask Marcus for separate approval of an exact small CPU experiment.",
        "",
        "**No GO is implied by green CI or Codex review. Physical energy: NOT_MEASURED.**",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="CL-9E static NO_TRAINING summary")
    parser.add_argument("--summary", action="store_true", required=True)
    parser.parse_args()
    print(summary_markdown(snapshot()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
