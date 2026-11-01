"""CL-6G pure numeric-difference diagnostics; no model imports or execution.

No model construction, inference, training, data generation or threshold approval.
Only scalar sequences supplied by a caller are compared.
"""
from __future__ import annotations

import math
from collections.abc import Sequence


def diagnose_pair(left: Sequence[float], right: Sequence[float]) -> dict:
    """Describe differences without inferring scientific equivalence.

    Relative difference is abs(a-b)/max(abs(a), abs(b)) and is defined as 0
    for exact zero pairs; this is deliberately scale-sensitive near zero.
    Tolerances, quality gates and acceptance criteria are NOT supplied here.
    """
    if not isinstance(left, (tuple, list)) or not isinstance(right, (tuple, list)):
        raise ValueError("both inputs must be finite numeric lists or tuples")
    if not left or len(left) != len(right):
        raise ValueError("nonempty, equal-length scalar sequences required")
    pairs = []
    for a, b in zip(left, right):
        if (isinstance(a, bool) or isinstance(b, bool)
                or not isinstance(a, (int, float))
                or not isinstance(b, (int, float))):
            raise ValueError("real scalar values required; booleans are not values")
        try:
            x, y = float(a), float(b)
        except (OverflowError, ValueError) as exc:
            raise ValueError("cannot represent scalar as finite float") from exc
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError("non-finite scalar is scientifically unverifiable")
        # Float conversion may round a valid integer (e.g., 2**53 + 1).
        # Preserve the integrity of the descriptive equality/count metrics.
        if (isinstance(a, int) and int(x) != a) or (isinstance(b, int) and int(y) != b):
            raise ValueError("integer value cannot be represented exactly as float")
        difference = abs(x - y)
        if not math.isfinite(difference):
            raise ValueError("difference overflow")
        scale = max(abs(x), abs(y))
        relative = difference / scale if scale else 0.0
        pairs.append((difference, relative))
    return {
        "kind": "CL6G_SCALAR_DIAGNOSTIC_ONLY",
        "scientific_decision": "NO_GO",
        "count": len(pairs),
        "exact_equal_count": sum(abs_delta == 0.0 for abs_delta, _ in pairs),
        "max_absolute_difference": max(delta for delta, _ in pairs),
        "max_symmetric_relative_difference": max(ratio for _, ratio in pairs),
        "relative_difference_definition": "abs(a-b)/max(abs(a),abs(b)); zero/zero := 0",
        "tolerance_approved": False,
        "equivalence_established": False,
        "energy_measured": False,
        "model_executed": False,
    }
