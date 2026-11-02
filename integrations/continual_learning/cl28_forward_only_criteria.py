"""CL-28 forward-only numeric acceptance review. NO MODEL execution/import.

Only synthetic, explicitly unauthenticated fixture records may be compared.
Passing this schema is not evidence that any real model has run.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
SPEC = HERE / "cl28_forward_only_criteria_candidate.json"
ARMS = ["local_hybrid", "sparse_local_hybrid", "sparse_shared_index"]
PAIRS = [["local_hybrid", "sparse_local_hybrid"],
         ["sparse_local_hybrid", "sparse_shared_index"]]
FIELDS = ["arm", "initial_weights", "hidden", "logits", "active_indices"]
MODEL_NAMES = ("cl2_mlp_baseline", "cl2_local_mlp",
               "cl2_sparse_forward", "cl2_shared_index_forward")


def require(ok: bool, reason: str):
    if not ok:
        raise ValueError("CL-28 NO_GO: " + reason)



def _exact_contract(actual: object, expected: object) -> bool:
    """Compare fixed contract values without bool/int/float coercion."""
    if type(actual) is not type(expected):
        return False
    if type(expected) is dict:
        return (actual.keys() == expected.keys()
                and all(_exact_contract(actual[key], item)
                        for key, item in expected.items()))
    if type(expected) is list:
        return (len(actual) == len(expected)
                and all(_exact_contract(a, b)
                        for a, b in zip(actual, expected)))
    return actual == expected


def validate_review_only(spec: dict) -> dict:
    require(type(spec) is dict and spec.get("kind") ==
            "CL28_FORWARD_ONLY_NUMERIC_ACCEPTANCE_CANDIDATE"
            and spec.get("version") == "cl28-review-v1"
            and spec.get("decision") == "NO_GO"
            and spec.get("review_only") is True, "execution status")
    require(_exact_contract(spec.get("sample"), {
        "seed": 101, "task": "T1", "partition": "train", "index": 0,
        "active_indices": [2, 8], "heldout_access": False,
    }), "frozen sample or active indices")
    require(_exact_contract(spec.get("arms"), ARMS) and _exact_contract(spec.get("pairs"), PAIRS), "paired arms")
    require(_exact_contract(spec.get("trace"), {
        "required_fields": FIELDS, "initial_weights_count": 154,
        "hidden_count": 8, "logits_count": 2, "finite_numeric_only": True,
        "same_initial_weights_required": True,
        "active_indices_exact_match": True, "no_loss_or_optimizer_fields": True,
    }), "forward trace contract")
    require(_exact_contract(spec.get("tolerance"), {
        "absolute": 1e-12, "relative": 1e-10,
        "method": "math_isclose_max_absolute_or_relative",
        "independently_approved": False,
    }), "candidate tolerance or false signoff")
    require(_exact_contract(spec.get("interpretation"), {
        "numeric_parity_only": True,
        "shared_index_reuse_exercised": False,
        "shared_index_reuse_requires_step_path": True,
        "algorithmic_efficiency_claim_authorized": False,
        "continual_learning_claim_authorized": False,
        "physical_energy_claim_authorized": False,
    }), "scientific interpretation scope")
    require(_exact_contract(spec.get("execution"), {
        "model_import_authorized": False, "inference_authorized": False,
        "training_authorized": False, "weight_updates_max": 0,
        "gradient_calls_max": 0, "forward_calls_max": 3,
        "observations_recorded": 0,
    }), "execution budget or false authorization")
    require(_exact_contract(spec.get("approvals"), {
        "scientific_signoff": False, "security_signoff": False,
        "marcus_exact_inference_authorization_reference": None,
    }) and type(spec.get("physical_energy")) is str
            and spec.get("physical_energy") == "NOT_MEASURED", "false approval/energy")
    require(not any(any(name in module for name in MODEL_NAMES) for module in sys.modules),
            "forbidden model import")
    return {
        "decision": "NO_GO", "scope": "FORWARD_ONLY_CRITERIA_PROPOSAL",
        "observations_authenticated": False, "models_imported": 0,
        "forward_calls_executed": 0, "training_steps": 0,
        "tolerance_approved": False, "physical_energy": "NOT_MEASURED",
        "numeric_parity_only": True,
        "shared_index_reuse_exercised": False,
    }


def _vector(value: object, count: int):
    require(type(value) is list and len(value) == count, "vector length")
    require(all(type(x) in (float, int) and math.isfinite(x) for x in value),
            "nonfinite/bool vector")
    return value


def audit_synthetic_fixture(spec: dict, record: object) -> dict:
    validate_review_only(spec)
    require(type(record) is dict and set(record) == {"synthetic_fixture", "traces"}
            and record["synthetic_fixture"] is True, "fixture provenance")
    traces = record["traces"]
    require(type(traces) is dict and set(traces) == set(ARMS), "three trace arms")
    for arm in ARMS:
        trace = traces[arm]
        require(type(trace) is dict and set(trace) == set(FIELDS)
                and trace["arm"] == arm, "trace shape")
        _vector(trace["initial_weights"], 154)
        _vector(trace["hidden"], 8)
        _vector(trace["logits"], 2)
        indices = trace["active_indices"]
        require(type(indices) is list and len(indices) == 2
                and all(type(i) is int for i in indices)
                and indices == [2, 8], "active index mismatch")
    base = traces[ARMS[0]]["initial_weights"]
    require(all(traces[arm]["initial_weights"] == base for arm in ARMS[1:]),
            "unequal initialization")
    first_mismatch = None
    checks = 0
    for left, right in PAIRS:
        for field in ("hidden", "logits"):
            for i, (a, b) in enumerate(zip(traces[left][field], traces[right][field])):
                checks += 1
                if first_mismatch is None and not math.isclose(
                    a, b, rel_tol=spec["tolerance"]["relative"],
                    abs_tol=spec["tolerance"]["absolute"],
                ):
                    first_mismatch = f"{left}/{right}:{field}[{i}]"
    return {
        "decision": "NO_GO", "scope": "SYNTHETIC_FIXTURE_ONLY",
        "candidate_parity_within_tolerance": first_mismatch is None,
        "first_mismatch": first_mismatch, "compared_values": checks,
        "observations_authenticated": False, "scientific_signoff": False,
        "model_inference_authorized": False, "physical_energy": "NOT_MEASURED",
        "numeric_parity_only": True,
        "shared_index_reuse_exercised": False,
    }


if __name__ == "__main__":
    print(json.dumps(validate_review_only(
        json.loads(SPEC.read_text(encoding="ascii"))), sort_keys=True))
