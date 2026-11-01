"""CL-5I AST-only review of generator instrumentation sites (NO DATA)."""
from __future__ import annotations

import ast
import hashlib
from pathlib import Path

from integrations.continual_learning.cl3y_readiness_audit import scientific_readiness_audit

GENERATOR = Path(__file__).with_name("cl2_shared_rule_candidate.py")


def static_generator_probe_audit() -> dict:
    """Identify exactly where future counters would attach, without executing code.

    Fail closed if the reviewed attempt loop or decision ordering changes.
    This is not an implemented generator instrumentation or an empirical run.
    """
    source = GENERATOR.read_bytes()
    module = ast.parse(source, filename=str(GENERATOR))
    funcs = [n for n in module.body if isinstance(n, ast.FunctionDef)
             and n.name == "generate_shared_rule_candidate"]
    if len(funcs) != 1:
        raise ValueError("generator entrypoint changed")
    loops = [n for n in ast.walk(funcs[0]) if isinstance(n, ast.For)
             and isinstance(n.iter, ast.Call)
             and isinstance(n.iter.func, ast.Name)
             and n.iter.func.id == "range"
             and len(n.iter.args) == 1
             and isinstance(n.iter.args[0], ast.Constant)
             and n.iter.args[0].value == 1_000_000]
    if len(loops) != 1:
        raise ValueError("expected exactly one bounded proposal loop")
    loop = loops[0]
    expected = [
        "if not any(remaining):\n    break",
        "features = tuple((int(rng.random() < probability) for _ in range(INPUT_SIZE)))",
        "label = global_label(features)",
        "if remaining[label] == 0 or features in used:\n    continue",
        "used.add(features)",
        "records.append(Example(features, label))",
        "remaining[label] -= 1",
    ]
    if len(loop.body) != len(expected):
        raise ValueError("proposal-loop statement count changed")
    def normalized(statement: str) -> str:
        return ast.dump(ast.parse(statement).body[0], include_attributes=False)
    for index, (actual, expected_source) in enumerate(zip(loop.body, expected)):
        if ast.dump(actual, include_attributes=False) != normalized(expected_source):
            raise ValueError(f"proposal-loop statement {index} differs from reviewed contract")
    if (sum(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
            and n.func.attr == "random" for n in ast.walk(loop)) != 1):
        raise ValueError("unexpected random draw call structure")
    readiness = scientific_readiness_audit()
    if readiness["decision"] != "NO_GO" or readiness["open_count"] != 9:
        raise ValueError("scientific gates unexpectedly changed")
    return {
        "kind": "CL5I_STATIC_INSTRUMENTATION_SITE_AUDIT_ONLY",
        "decision": "NO_GO",
        "generator_source_sha256_observed": hashlib.sha256(source).hexdigest(),
        "attempt_loop_cap_per_partition": 1_000_000,
        "reviewed_statement_order": [
            "quota_complete_early_break",
            "propose_feature_vector",
            "compute_global_xor_label",
            "class_full_or_duplicate_skip",
            "mark_globally_used",
            "append_accepted_example",
            "decrement_class_quota",
        ],
        "proposed_future_counters": [
            "candidate_attempts",
            "class_full_rejections",
            "duplicate_rejections_after_class_check",
            "accepted_records",
            "cap_exhaustion",
        ],
        "ambiguity": "class-full check short-circuits duplicate check; preserve priority",
        "instrumentation_implemented": False,
        "generator_modified": False,
        "dataset_generation_executed": False,
        "model_training_executed": False,
        "energy_measured": False,
        "scientific_gates_open": 9,
    }
