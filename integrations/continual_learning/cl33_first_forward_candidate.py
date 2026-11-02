"""CL-33 proposed three-forward numerical probe, deliberately HARD NO_GO.

Imports no model at import/CLI/preflight time. No inference workflow invokes
execute_proposed_forward. A later reviewed source change AND Marcus's separate
exact authorization are needed before any model execution.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ARMS = ("local_hybrid", "sparse_local_hybrid", "sparse_shared_index")
ACTIVE = [2, 8]
WEIGHTS = 154
HIDDEN = 8
LOGITS = 2
MAX_RESULT_BYTES = 1_048_576
ABS_TOL = 1e-12
REL_TOL = 1e-10


class InferenceNotAuthorized(RuntimeError):
    pass


def _separate_exact_authorization_present() -> bool:
    # Neither CLI inputs nor environment variables can unlock this candidate.
    # A future source-reviewed change is required. CURRENTLY ALWAYS FALSE.
    return False


def review_only() -> dict:
    forbidden = ("cl2_mlp_baseline", "cl2_local_mlp",
                 "cl2_sparse_forward", "cl2_shared_index_forward")
    if any(any(fragment in module for fragment in forbidden) for module in sys.modules):
        raise RuntimeError("CL-33 NO_GO: model imported during preflight")
    return {
        "decision": "NO_GO",
        "sample": {"seed": 101, "task": "T1", "partition": "train", "index": 0},
        "model_arms": list(ARMS),
        "forward_calls_executed": 0,
        "model_imports": 0,
        "training_steps": 0,
        "weight_updates": 0,
        "gradient_calls": 0,
        "physical_energy": "NOT_MEASURED",
        "candidate_tolerance": {"absolute": ABS_TOL, "relative": REL_TOL,
                                "scientifically_approved": False},
        "owner_exact_inference_authorization_present": False,
        "risk_acceptance_recorded": False,
        "image_oci_digest_approved": False,
    }


def _flatten_initial_weights(model: object) -> list[float]:
    return ([value for row in model.w1 for value in row]
            + list(model.b1)
            + [value for row in model.w2 for value in row]
            + list(model.b2))


def execute_proposed_forward() -> dict:
    """Future scope only. NEVER called by CL-33 CI or workflow."""
    if not _separate_exact_authorization_present():
        raise InferenceNotAuthorized(
            "CL-33 HARD NO_GO: exact owner inference authorization, accepted "
            "residual risks and final image identity are absent"
        )

    # Unreachable in the reviewed CL-33 revision. Reviewable future design:
    # No model import occurs unless a later approved source change unlocks it.
    sys.path.insert(0, "/opt/research")
    from integrations.continual_learning.cl2_local_mlp import LocalHebbianMLP
    from integrations.continual_learning.cl2_sparse_forward import SparseForwardLocalMLP
    from integrations.continual_learning.cl2_shared_index_forward import SharedIndexSparseForwardLocalMLP

    packet = json.loads((HERE / "cl25_single_archived_input.json").read_text(encoding="ascii"))
    expected = [0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    if (packet.get("seed") != 101 or packet.get("task") != "T1"
            or packet.get("partition") != "train" or type(packet.get("index")) is not int
            or packet["index"] != 0 or packet.get("features") != expected
            or packet.get("heldout_data_packaged") is not False
            or packet.get("decision") != "NO_GO"):
        raise RuntimeError("CL-33 NO_GO: frozen sample drift")
    features = tuple(expected)
    active = [i for i, bit in enumerate(features) if bit]
    if active != ACTIVE:
        raise RuntimeError("CL-33 NO_GO: active index drift")

    models = (
        LocalHebbianMLP(seed=101),
        SparseForwardLocalMLP(seed=101),
        SharedIndexSparseForwardLocalMLP(seed=101),
    )
    traces = {}
    for arm, model in zip(ARMS, models):
        if model.optimizer_steps != 0 or model.parameter_count != WEIGHTS:
            raise RuntimeError("CL-33 NO_GO: model state/architecture drift")
        before = _flatten_initial_weights(model)
        hidden, logits = model._forward(model._input(features))
        after = _flatten_initial_weights(model)
        if (len(before) != WEIGHTS or before != after or
                len(hidden) != HIDDEN or len(logits) != LOGITS or
                not all(type(value) in (float, int) and math.isfinite(value)
                        for value in before + hidden + logits) or model.optimizer_steps != 0):
            raise RuntimeError("CL-33 NO_GO: unexpected mutation/shape/nonfinite value")
        traces[arm] = {"arm": arm, "initial_weights": before,
                       "hidden": hidden, "logits": logits,
                       "active_indices": active}

    baseline = traces[ARMS[0]]["initial_weights"]
    if any(traces[arm]["initial_weights"] != baseline for arm in ARMS[1:]):
        raise RuntimeError("CL-33 NO_GO: unequal initial weights")
    comparisons = []
    for left, right in ((ARMS[0], ARMS[1]), (ARMS[1], ARMS[2])):
        for field in ("hidden", "logits"):
            for index, (a, b) in enumerate(zip(traces[left][field],
                                                traces[right][field])):
                comparisons.append({"pair": [left, right], "field": field,
                                    "index": index,
                                    "within_candidate_tolerance": math.isclose(
                                        a, b, abs_tol=ABS_TOL, rel_tol=REL_TOL)})
    if len(comparisons) != 20:
        raise RuntimeError("CL-33 NO_GO: numerical comparison count drift")
    result = {"scope": "EXPLORATORY_NUMERIC_PARITY_ONLY",
              "seed": 101, "task": "T1", "index": 0,
              "traces": traces, "comparisons": comparisons,
              "forward_calls": 3, "training_steps": 0, "weight_updates": 0,
              "physical_energy": "NOT_MEASURED",
              "scientifically_approved_tolerance": False,
              "host_security_independently_certified": False}
    if len(json.dumps(result, allow_nan=False).encode("utf-8")) > MAX_RESULT_BYTES:
        raise RuntimeError("CL-33 NO_GO: oversized output")
    return result


if __name__ == "__main__":
    # CLI always performs preflight only, including on manual workflow_dispatch.
    print(json.dumps(review_only(), sort_keys=True))
