#!/usr/bin/env python3
"""CL-46 archive-only verification. Standard library; NO MODEL / NO TRAINING."""
import hashlib
import itertools
import json
import math
from pathlib import Path

ARCHIVE = (
    Path(__file__).resolve().parents[2]
    / "experiments/cl46_first_one_sample_forward/cl45-one-sample-results.json"
)
EXPECTED_SHA256 = "33e9f3f03e2854c7ed0e80859f5d97035bebd3cee95ef98a86eb01ce42c754cf"
IMAGE_CONFIG_CLAIM = "sha256:34b4fb59817acf1a5a5a8318ef34a26b73b691c0bacf2b69666dcb04efdce2ff"
ARMS = ("local_hybrid", "sparse_local_hybrid", "sparse_shared_index")
APPROVAL = "MARCUS-EXACT-ONE-SAMPLE-SEED101-T1-TRAIN0-THREE-FORWARDS"


def validate(data: bytes) -> None:
    assert hashlib.sha256(data).hexdigest() == EXPECTED_SHA256, "Archive byte hash differs"
    receipt = json.loads(data)
    assert receipt["kind"] == "CL36_EXPLORATORY_FORWARD_ONLY_OBSERVATION"
    assert receipt["approval_reference"] == APPROVAL
    assert receipt["image_config_id_claim"] == IMAGE_CONFIG_CLAIM
    assert (receipt["seed"], receipt["task"], receipt["partition"], receipt["index"]) == (101, "T1", "train", 0)
    for field, expected in (
        ("forward_calls", 3),
        ("training_steps", 0),
        ("gradient_calls", 0),
        ("weight_updates", 0),
        ("compared_values", 20),
    ):
        assert type(receipt[field]) is int and receipt[field] == expected, field
    assert receipt["physical_energy"] == "NOT_MEASURED"
    assert receipt["numeric_parity_only"] is True
    assert receipt["candidate_parity_within_tolerance"] is True
    assert receipt["first_mismatch"] is None
    assert receipt["shared_index_reuse_exercised"] is False
    assert receipt["host_isolation_independently_certified"] is False
    tolerance = receipt["tolerance"]
    assert tolerance == {"absolute": 1e-12, "relative": 1e-10, "independently_approved": False}
    traces = receipt["traces"]
    assert set(traces) == set(ARMS)
    for arm in ARMS:
        trace = traces[arm]
        assert trace["arm"] == arm
        assert trace["active_indices"] == [2, 8]
        for key, expected_length in (("initial_weights", 154), ("hidden", 8), ("logits", 2)):
            values = trace[key]
            assert isinstance(values, list) and len(values) == expected_length
            assert all(type(v) in (int, float) and math.isfinite(v) for v in values)
    baseline_weights = traces[ARMS[0]]["initial_weights"]
    assert all(traces[arm]["initial_weights"] == baseline_weights for arm in ARMS[1:])
    for left, right in itertools.combinations(ARMS, 2):
        for field in ("hidden", "logits"):
            first, second = traces[left][field], traces[right][field]
            assert all(a == b for a, b in zip(first, second)), (left, right, field)
    print("CL-46 ARCHIVE VERIFIED: exact SHA-256, scope, counters, 3-arm numerical parity")
    print("NO MODEL IMPORTED; NO INFERENCE; NO TRAINING; NO ENERGY MEASUREMENT; HARD NO_GO")


if __name__ == "__main__":
    validate(ARCHIVE.read_bytes())
