"""CL-36 proposed exactly-one-sample/three-forward runner.

DEFAULT AND ALL CI BEHAVIOR: HARD NO_GO. No model import or inference.
A future separately reviewed GitHub Actions job may invoke --approved-once only
AFTER Marcus gives an exact start approval referencing the final image ID.
Environment markers are audit metadata, NOT cryptographic authorization;
the externally disabled workflow and explicit human approval are the gate.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import sys

RESEARCH = Path("/opt/research/integrations/continual_learning")
ARMS = ("local_hybrid", "sparse_local_hybrid", "sparse_shared_index")
FEATURES = (0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0)
ABS_TOL = 1e-12
REL_TOL = 1e-10


def no_go_receipt() -> dict:
    return {"decision": "HARD_NO_GO", "models_imported": 0,
            "forward_calls_executed": 0, "training_steps": 0,
            "gradient_calls": 0, "weight_updates": 0,
            "physical_energy": "NOT_MEASURED",
            "owner_exact_start_approved": False,
            "approved_oci_manifest": None}


def _flatten(model: object) -> list[float]:
    return ([n for row in model.w1 for n in row] + list(model.b1)
            + [n for row in model.w2 for n in row] + list(model.b2))


def run_one_approved_probe() -> dict:
    """Not invoked by any CL-36 GitHub Actions step."""
    approval = os.environ.get("CL36_OWNER_APPROVAL_REFERENCE", "")
    image_id = os.environ.get("CL36_FINAL_IMAGE_CONFIG_ID", "")
    if (not approval.startswith("MARCUS-EXACT-ONE-SAMPLE-")
            or not image_id.startswith("sha256:")
            or len(image_id) != 71):
        raise RuntimeError("CL-36 HARD NO_GO: missing exact start/identity evidence")

    # Source/model imports occur ONLY after the explicit CLI + environment gate.
    sys.path.insert(0, "/opt/research")
    from integrations.continual_learning.cl2_local_mlp import LocalHebbianMLP
    from integrations.continual_learning.cl2_sparse_forward import SparseForwardLocalMLP
    from integrations.continual_learning.cl2_shared_index_forward import SharedIndexSparseForwardLocalMLP

    input_packet = json.loads((RESEARCH / "cl25_single_archived_input.json").read_text(encoding="ascii"))
    if (type(input_packet) is not dict
            or input_packet.get("decision") != "NO_GO"
            or type(input_packet.get("seed")) is not int or input_packet["seed"] != 101
            or input_packet.get("task") != "T1"
            or input_packet.get("partition") != "train"
            or type(input_packet.get("index")) is not int or input_packet["index"] != 0
            or input_packet.get("features") != list(FEATURES)
            or input_packet.get("heldout_data_packaged") is not False):
        raise RuntimeError("CL-36 NO_GO: frozen single sample drift")
    active = [i for i, value in enumerate(FEATURES) if value]
    if active != [2, 8]:
        raise RuntimeError("CL-36 NO_GO: active input drift")

    constructors = (LocalHebbianMLP, SparseForwardLocalMLP,
                    SharedIndexSparseForwardLocalMLP)
    traces = {}
    calls = 0
    for arm, Model in zip(ARMS, constructors):
        model = Model(seed=101)
        if model.optimizer_steps != 0 or model.parameter_count != 154:
            raise RuntimeError("CL-36 NO_GO: unexpected model state")
        before = _flatten(model)
        if len(before) != 154:
            raise RuntimeError("CL-36 NO_GO: initial weight count")
        hidden, logits = model._forward(model._input(FEATURES))
        calls += 1
        if (len(hidden) != 8 or len(logits) != 2
                or _flatten(model) != before or model.optimizer_steps != 0
                or not all(type(x) in (int, float) and math.isfinite(x)
                           for x in before + hidden + logits)):
            raise RuntimeError("CL-36 NO_GO: output/weight integrity")
        traces[arm] = {"arm": arm, "initial_weights": before,
                       "hidden": hidden, "logits": logits,
                       "active_indices": active}
    if calls != 3:
        raise RuntimeError("CL-36 NO_GO: incorrect forward count")
    initial = traces[ARMS[0]]["initial_weights"]
    if any(traces[name]["initial_weights"] != initial for name in ARMS[1:]):
        raise RuntimeError("CL-36 NO_GO: unequal initialization")

    comparison_count = 0
    mismatches = []
    for left, right in ((ARMS[0], ARMS[1]), (ARMS[1], ARMS[2])):
        for field in ("hidden", "logits"):
            for index, (a, b) in enumerate(zip(traces[left][field],
                                                traces[right][field])):
                comparison_count += 1
                if not math.isclose(a, b, abs_tol=ABS_TOL, rel_tol=REL_TOL):
                    mismatches.append({"pair": [left, right],
                                       "field": field, "index": index})
    if comparison_count != 20:
        raise RuntimeError("CL-36 NO_GO: comparison count drift")

    result = {
        "kind": "CL36_EXPLORATORY_FORWARD_ONLY_OBSERVATION",
        "approval_reference": approval, "image_config_id_claim": image_id,
        "seed": 101, "task": "T1", "partition": "train", "index": 0,
        "traces": traces, "compared_values": comparison_count,
        "candidate_parity_within_tolerance": len(mismatches) == 0,
        "first_mismatch": mismatches[0] if mismatches else None,
        "tolerance": {"absolute": ABS_TOL, "relative": REL_TOL,
                      "independently_approved": False},
        "numeric_parity_only": True, "shared_index_reuse_exercised": False,
        "forward_calls": calls, "training_steps": 0,
        "gradient_calls": 0, "weight_updates": 0,
        "physical_energy": "NOT_MEASURED",
        "host_isolation_independently_certified": False,
    }
    serialized = json.dumps(result, allow_nan=False, sort_keys=True)
    if len(serialized.encode("utf-8")) > 1_048_576:
        raise RuntimeError("CL-36 NO_GO: result too large")
    return result


if __name__ == "__main__":
    if sys.argv[1:] != ["--approved-once"]:
        print(json.dumps(no_go_receipt(), sort_keys=True))
    else:
        # This explicit mode is NEVER invoked by the CL-36 workflow.
        print(json.dumps(run_one_approved_probe(), allow_nan=False, sort_keys=True))
