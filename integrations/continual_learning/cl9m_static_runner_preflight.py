"""CL-9M source-only H1 runner preflight. NEVER returns execution GO.

The checker verifies a narrow audited static workflow by conservative textual
matching. It cannot infer effective GitHub settings or prove network isolation.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STATIC_WORKFLOW = ROOT / ".github/workflows/cl9i-static-h1.yml"
CL9L = Path(__file__).resolve().parent / "cl9l_bounded_h1_runner_security_profile.json"
READ_ONLY_RULES = (
    r"(?m)^permissions:\s*\n\s+contents:\s*read\s*$",
    r"(?m)^\s+timeout-minutes:\s*3\s*$",
    r"(?m)^\s+persist-credentials:\s*false\s*$",
    r"(?m)^\s+python-version:\s*[\"']?3\.11[\"']?\s*$",
    r"(?m)^\s+python -B cl9i_static_h1_validator\.py\s*$",
    r"(?m)^\s+python -B -m unittest -v test_cl9i_static_h1_validator\.py\s*$",
)


def require(value: bool, message: str) -> None:
    if not value:
        raise ValueError(f"CL-9M NO_GO: {message}")


def assess(workflow: str, profile: dict) -> dict:
    """Report bounded source findings only. Never infer execution permission."""
    require(type(workflow) is str and len(workflow) > 0, "workflow missing")
    require(type(profile) is dict and
            profile.get("kind") == "CL9L_BOUNDED_H1_RUNNER_SECURITY_PROFILE" and
            profile.get("decision") == "NO_GO" and
            profile.get("review_only") is True and
            profile.get("runnable_h1_workflow_created") is False and
            profile.get("execution_authorized") is False and
            profile.get("training_authorized") is False and
            profile.get("inference_authorized") is False and
            profile.get("network_isolation_verified") is False,
            "CL-9L profile no longer explicitly denies execution")

    for pattern in READ_ONLY_RULES:
        require(re.search(pattern, workflow) is not None,
                f"static workflow required source pattern missing: {pattern}")
    # This is an intentional exact-content allowlist, not a general-purpose YAML
    # parser or proof about jobs outside CL-9I. Unknown lines fail closed.
    expected = '''name: CL-9I Static H1 Method Contract

on:
  pull_request:
    paths:
      - "integrations/continual_learning/**"
      - ".github/workflows/cl9i-static-h1.yml"
  push:
    branches: [main]
    paths:
      - "integrations/continual_learning/**"
      - ".github/workflows/cl9i-static-h1.yml"
  workflow_dispatch:

permissions:
  contents: read

jobs:
  static-contract:
    runs-on: ubuntu-latest
    timeout-minutes: 3
    steps:
      - uses: actions/checkout@v4
        with:
          persist-credentials: false
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - name: Validate static CL-9I contract
        working-directory: integrations/continual_learning
        run: |
          python -B cl9i_static_h1_validator.py
          python -B -m unittest -v test_cl9i_static_h1_validator.py
'''
    require(workflow.replace("\r\n", "\n") == expected,
            "CL-9I workflow differs from reviewed source allowlist")
    # Do not claim this runner is offline: GitHub-hosted egress defaults to on.
    return {
        "decision": "NO_GO",
        "scope": "CL9I_STATIC_SOURCE_ONLY",
        "source_checks": "PASS",
        "model_training_authorized": False,
        "model_inference_authorized": False,
        "network_isolation_verified": False,
        "github_admin_settings_verified": False,
        "future_h1_workflow_allowed": False,
        "physical_energy": "NOT_MEASURED",
    }


def main() -> None:
    workflow = STATIC_WORKFLOW.read_text(encoding="utf-8")
    profile = json.loads(CL9L.read_text(encoding="utf-8"))
    print(json.dumps(assess(workflow, profile), sort_keys=True))


if __name__ == "__main__":
    main()
