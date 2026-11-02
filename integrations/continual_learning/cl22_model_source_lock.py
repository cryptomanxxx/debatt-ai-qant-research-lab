"""CL-22 read-only source closure audit. No model imports or execution."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SPEC = HERE / "cl22_model_source_lock.json"
EXPECTED_FILES = frozenset({
    "cl1_cpu.py",
    "cl2_mlp_baseline.py",
    "cl2_local_mlp.py",
    "cl2_sparse_forward.py",
    "cl2_shared_index_forward.py",
    "cl10j_source_event_collector.py",
    "cl10k_weight_update_hooks.py",
    "cl10l_forward_hooks.py",
    "cl10m_training_hooks.py",
})
PREFIX = "integrations/continual_learning/"
ARMS = ["local_hybrid", "sparse_local_hybrid", "sparse_shared_index"]


def validate_review_only(spec: dict, source_bytes: dict[str, bytes]) -> dict:
    if (type(spec) is not dict
            or spec.get("kind") != "CL22_MODEL_SOURCE_CLOSURE_REVIEW_ONLY"
            or spec.get("decision") != "NO_GO"
            or spec.get("review_only") is not True
            or spec.get("model_arms") != ARMS
            or spec.get("final_model_image_digest") is not None
            or any(spec.get(k) is not False for k in (
                "model_import_authorized", "inference_authorized",
                "training_authorized", "energy_measurement_authorized",
                "security_signoff", "scientific_signoff"))):
        raise ValueError("CL-22 NO_GO: authorization or model scope drift")
    frozen = spec.get("source_git_blob_sha1")
    expected_paths = {PREFIX + name for name in EXPECTED_FILES}
    if (type(frozen) is not dict or set(frozen) != expected_paths
            or set(source_bytes) != expected_paths):
        raise ValueError("CL-22 NO_GO: model source closure drift")
    for path in sorted(expected_paths):
        data = source_bytes[path]
        if type(data) is not bytes:
            raise ValueError("CL-22 NO_GO: source must be raw bytes")
        blob = b"blob " + str(len(data)).encode("ascii") + b"\\0" + data
        if hashlib.sha1(blob).hexdigest() != frozen[path]:
            raise ValueError("CL-22 NO_GO: source fingerprint drift: " + path)
        # Parse source text only. Never import or construct any model.
        tree = ast.parse(data, filename=path)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                modules = [node.module] if node.module else []
            elif isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            else:
                continue
            for module in modules:
                prefix = "integrations.continual_learning."
                if module.startswith(prefix):
                    dependency = PREFIX + module.removeprefix(prefix) + ".py"
                    if dependency not in expected_paths:
                        raise ValueError("CL-22 NO_GO: unfrozen model dependency: " + dependency)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in {
                "__import__", "eval", "exec",
            }:
                raise ValueError("CL-22 NO_GO: dynamic evaluation in source: " + path)
    return {
        "decision": "NO_GO",
        "source_files_pinned": len(expected_paths),
        "source_closure_audited": True,
        "final_model_image_pinned": False,
        "model_imports_executed": 0,
        "inference_authorized": False,
        "training_authorized": False,
        "physical_energy": "NOT_MEASURED",
    }


def audit_current_files() -> dict:
    spec = json.loads(SPEC.read_text(encoding="ascii"))
    paths = spec["source_git_blob_sha1"]
    return validate_review_only(spec, {
        path: (ROOT / path).read_bytes() for path in paths
    })


if __name__ == "__main__":
    print(json.dumps(audit_current_files(), sort_keys=True))
