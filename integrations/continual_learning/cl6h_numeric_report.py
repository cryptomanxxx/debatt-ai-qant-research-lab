"""CL-6H render-only diagnostics; no model imports, execution or files written."""
from __future__ import annotations

import math
from collections.abc import Mapping


FIELDS = ("w1", "b1", "w2", "b2")
PAIRS = ("primary_sparse", "secondary_index_reuse")


def render_numeric_report(pair_id: str, diagnostics: Mapping[str, dict]) -> str:
    """Render previously supplied CL-6G diagnostics as an explicitly NO_GO report.

    Only the four parameter fields are accepted; optimizer/counter/hyperparameter
    state and numerical tolerances are NOT verified by this reporter.
    """
    if pair_id not in PAIRS:
        raise ValueError("unknown preregistered paired comparison")
    if not isinstance(diagnostics, Mapping) or set(diagnostics) != set(FIELDS):
        raise ValueError("all four exact parameter diagnostics required")
    lines = [
        "## CL-6H — numeric difference summary (DIAGNOSTIC ONLY)",
        "",
        "> **NO_GO / NOT A MODEL EXPERIMENT.** Rendered records only; no tolerance, "
        "causal equivalence, model quality or physical energy is approved.",
        "",
        f"Paired comparison: \`{pair_id}\`",
        "",
        "| Parameter field | Scalars | Exact equal | Max abs delta | Max symmetric relative delta |",
        "|---|---:|---:|---:|---:|",
    ]
    for field in FIELDS:
        item = diagnostics[field]
        if not isinstance(item, Mapping):
            raise ValueError("diagnostic must be a mapping")
        if (item.get("kind") != "CL6G_SCALAR_DIAGNOSTIC_ONLY"
                or item.get("scientific_decision") != "NO_GO"
                or item.get("tolerance_approved") is not False
                or item.get("equivalence_established") is not False
                or item.get("energy_measured") is not False
                or item.get("model_executed") is not False):
            raise ValueError("diagnostic has an unapproved scientific status")
        count, equal = item.get("count"), item.get("exact_equal_count")
        if (type(count) is not int or type(equal) is not int
                or count <= 0 or not 0 <= equal <= count):
            raise ValueError("invalid diagnostic counts")
        absolute = item.get("max_absolute_difference")
        relative = item.get("max_symmetric_relative_difference")
        if (type(absolute) not in (float, int) or type(relative) not in (float, int)
                or not math.isfinite(absolute) or not math.isfinite(relative)
                or absolute < 0 or not 0 <= relative <= 2):
            raise ValueError("invalid diagnostic magnitude")
        if (absolute == 0) != (equal == count):
            raise ValueError("inconsistent exact-equality summary")
        lines.append(f"| \`{field}\` | {count} | {equal} | {absolute:.9g} | {relative:.9g} |")
    lines += [
        "",
        "**Scope limitation:** This partial four-tensor view does not validate learning "
        "rates, optimizer steps, local counters, input/seed parity, numerical "
        "tolerances, hidden states or post-update learning trajectories.",
        "",
        "**No physical energy, memory traffic, Loihi, Hala Point or Q.ANT hardware performance measured.**",
        "",
        "**All CL-6A gates remain OPEN; Marcus's separate approval is required before training.**",
        "",
    ]
    return "\n".join(lines)
