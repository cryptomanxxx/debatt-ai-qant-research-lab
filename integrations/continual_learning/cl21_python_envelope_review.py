"""CL-21 audit candidate Python base digest and Docker envelope without model."""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SPEC = HERE / "cl21_pinned_python_envelope_candidate.json"
BASE = "python:3.11-slim@sha256:673719e071ecf27434fd2635bbd8d00f49f4fdd81f243143438a42ced8bedf60"


def validate_review_only(spec: dict, docker: str, workflow: str) -> dict:
    if (type(spec) is not dict
            or spec.get("kind") != "CL21_PINNED_PYTHON_NO_MODEL_ENVELOPE_CANDIDATE"
            or spec.get("decision") != "NO_GO"
            or spec.get("review_only") is not True
            or spec.get("python_base_image") != BASE
            or spec.get("from_verified_cl20_run_id") != 41455392294):
        raise ValueError("CL-21 NO_GO: image/proposal drift")
    bound = spec.get("container_envelope", {})
    if bound != {
        "network": "none", "capabilities": "ALL_DROPPED",
        "no_new_privileges": True, "read_only": True,
        "uid_gid": "65534:65534", "pids_limit": 32,
        "memory_mib": 128, "cpus": 1,
        "job_timeout_minutes": 10, "pull_policy": "never",
    }:
        raise ValueError("CL-21 NO_GO: resources drift")
    numeric = spec.get("future_numeric_sample_budget", {})
    if numeric != {
        "seeds": [101], "samples": 1, "model_instances_max": 3,
        "forward_calls_max": 3, "weight_updates_max": 0,
        "training_steps_max": 0, "physical_energy": "NOT_MEASURED",
    }:
        raise ValueError("CL-21 NO_GO: future experiment budget drift")
    if any(spec.get(field) is not False for field in (
        "runtime_boundary_approved", "final_image_digest_approved",
        "model_inference_authorized", "training_authorized",
        "energy_measurement_authorized", "scientific_signoff",
        "security_signoff")) or spec.get("marcus_exact_inference_authorization_reference") is not None:
        raise ValueError("CL-21 NO_GO: false approval claims")
    if ("FROM " + BASE) not in docker:
        raise ValueError("CL-21 NO_GO: Docker base digest unpinned")
    if 'ENTRYPOINT ["python3", "-I", "-B", "/opt/research/integrations/continual_learning/cl20_no_model_container_probe.py"]' not in docker:
        raise ValueError("CL-21 NO_GO: model entrypoint drift")
    # The workflow is tiny and security-critical. Pin the *whole executable
    # job*, not substrings anywhere in YAML: comments or a second timeout
    # must never satisfy the checks for the real Docker run step.
    expected_job = """  no_model_network_probe:
    runs-on: ubuntu-latest
    timeout-minutes: 10
    steps:
      - uses: actions/checkout@v5
      - name: Build Python candidate with project sources (NO MODEL IMPORT)
        run: docker build --file integrations/continual_learning/Dockerfile.cl20-probe --tag local/cl20-no-model-probe:review .
      - name: Record local candidate image ID (NOT final approved digest)
        run: docker image inspect --format '{{.Id}}' local/cl20-no-model-probe:review
      - name: Prove this unprivileged read-only container has no outbound TCP (NO MODEL)
        run: |
          docker run --rm --pull=never --network=none --cap-drop=ALL --security-opt=no-new-privileges --read-only --pids-limit=32 --memory=128m --cpus=1 --user=65534:65534 local/cl20-no-model-probe:review"""
    if (workflow.count("\njobs:\n") != 1
            or workflow.split("\njobs:\n", 1)[1].strip() != expected_job.strip()):
        raise ValueError("CL-21 NO_GO: executable CI job or resource budget drift")
    if "permissions:\n  contents: read\n" not in workflow:
        raise ValueError("CL-21 NO_GO: workflow permission drift")
    return {
        "decision": "NO_GO",
        "python_base_manifest_pinned": True,
        "final_model_image_pinned": False,
        "container_probe_only": True,
        "model_inference_authorized": False,
        "training_authorized": False,
        "physical_energy": "NOT_MEASURED",
    }


def audit_current_files() -> dict:
    return validate_review_only(
        json.loads(SPEC.read_text(encoding="ascii")),
        (HERE / "Dockerfile.cl20-probe").read_text(encoding="utf-8"),
        (ROOT / ".github/workflows/cl20-no-model-container.yml").read_text(encoding="utf-8"),
    )


if __name__ == "__main__":
    print(json.dumps(audit_current_files(), sort_keys=True))
