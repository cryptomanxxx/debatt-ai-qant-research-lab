"""CL-6D initialization source contract: AST-only, NO MODEL EXECUTION."""
from __future__ import annotations

import argparse
import ast
import hashlib
from pathlib import Path

from integrations.continual_learning.cl6b_source_arm_equivalence import PATHS, audit_sources
from integrations.continual_learning.cl6a_readiness_review import PATH as CL6A_PATH, review
import json


EXPECTED_BASE_FIELDS = {"w1", "b1", "w2", "b2", "learning_rate", "optimizer_steps"}
EXPECTED_LOCAL_FIELDS = {"local_rate", "logical_counters"}


def _class(source: str, name: str) -> ast.ClassDef:
    found = [
        n for n in ast.parse(source).body
        if isinstance(n, ast.ClassDef) and n.name == name
    ]
    if len(found) != 1:
        raise ValueError(f"{name}: exact class required")
    return found[0]


def _init(cls: ast.ClassDef) -> ast.FunctionDef | None:
    found = [
        n for n in cls.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == "__init__"
    ]
    if len(found) > 1:
        raise ValueError("duplicate initializer")
    return found[0] if found else None


def _fields(method: ast.FunctionDef) -> set[str]:
    names = []
    for node in ast.walk(method):
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
            targets = [node.target]
        else:
            continue
        for target in targets:
            if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == "self":
                names.append(target.attr)
    if len(names) != len(set(names)):
        raise ValueError("duplicate model initialization assignment")
    return set(names)


def _root_name(target: ast.AST) -> str | None:
    while isinstance(target, (ast.Subscript, ast.Attribute)):
        target = target.value
    return target.id if isinstance(target, ast.Name) else None


def _guard_initializer(method: ast.FunctionDef, *, local: bool) -> None:
    # Only reviewed direct self.field assignments may initialize state.
    # No nested writes, in-place mutation, or rebinding forwarded arguments.
    for node in ast.walk(method):
        targets = ()
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign, ast.NamedExpr)):
            targets = (node.target,)
        elif isinstance(node, ast.Delete):
            targets = node.targets
        for target in targets:
            if local and _root_name(target) in {"seed", "learning_rate"}:
                raise ValueError("forwarded initializer argument rebound")
            if _root_name(target) == "self" and not (
                isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id == "self"
            ):
                raise ValueError("unreviewed nested or augmented state write")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if _root_name(node.func.value) == "self":
                raise ValueError("initializer calls mutating or unreviewed self operation")
            if not local and _root_name(node.func.value) == "rng" and ast.unparse(node.func) != "rng.uniform":
                raise ValueError("unreviewed RNG state operation")


def _field_values(method: ast.FunctionDef) -> dict[str, str]:
    return {
        node.targets[0].attr: ast.unparse(node.value)
        for node in ast.walk(method)
        if isinstance(node, ast.Assign) and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Attribute)
        and isinstance(node.targets[0].value, ast.Name)
        and node.targets[0].value.id == "self"
    }


def audit_initializers(sources: dict[str, str]) -> dict:
    # Fail if CL-6B's source routing contract no longer holds.
    parent = audit_sources(sources)
    if parent["decision"] != "NO_GO":
        raise ValueError("CL-6B source status changed")
    classes = {
        key: _class(sources[key], name)
        for key, name in (
            ("dense", "TinyDenseMLP"),
            ("local", "LocalHebbianMLP"),
            ("sparse", "SparseForwardLocalMLP"),
            ("shared", "SharedIndexSparseForwardLocalMLP"),
        )
    }
    base = _init(classes["dense"])
    local = _init(classes["local"])
    if base is None or local is None:
        raise ValueError("base/local initializer missing")
    if _init(classes["sparse"]) is not None or _init(classes["shared"]) is not None:
        raise ValueError("sparse subclasses must inherit local initialization")
    # Method discovery alone misses class-level assignment rebinding __init__.
    for key in ("dense", "local"):
        for member in classes[key].body:
            if isinstance(member, ast.FunctionDef):
                continue
            if (isinstance(member, ast.Expr) and isinstance(member.value, ast.Constant)
                    and isinstance(member.value.value, str)):
                continue
            raise ValueError(f"{key}: unreviewed class-level initializer binding")
    _guard_initializer(base, local=False)
    _guard_initializer(local, local=True)
    if _fields(base) != EXPECTED_BASE_FIELDS or _fields(local) != EXPECTED_LOCAL_FIELDS:
        raise ValueError("model state initialization fields changed")
    base_text = ast.unparse(base)
    local_text = ast.unparse(local)
    values = _field_values(base)
    expected_weights = {
        "w1": "[[rng.uniform(-0.2, 0.2) for _ in range(INPUT_SIZE)] for _ in range(HIDDEN_SIZE)]",
        "w2": "[[rng.uniform(-0.2, 0.2) for _ in range(HIDDEN_SIZE)] for _ in range(N_CLASSES)]",
    }
    if any(values.get(name) != expr for name, expr in expected_weights.items()):
        raise ValueError("both weight layers must use reviewed seeded RNG")
    rng_bindings = [
        node for node in ast.walk(base)
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "rng" for t in node.targets)
    ]
    if len(rng_bindings) != 1 or ast.unparse(rng_bindings[0].value) != "random.Random(seed)":
        raise ValueError("seeded RNG binding changed")
    if ("random.Random(seed)" not in base_text or
            "rng.uniform(-0.2, 0.2)" not in base_text or
            "self.b1 = [0.0] * HIDDEN_SIZE" not in base_text or
            "self.b2 = [0.0] * N_CLASSES" not in base_text or
            "self.optimizer_steps = 0" not in base_text):
        raise ValueError("base seeded initialization pattern changed")
    # Verify the *actual* super() call instead of just the presence of a string.
    super_calls = [
        n for n in ast.walk(local)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        and n.func.attr == "__init__"
        and isinstance(n.func.value, ast.Call)
        and isinstance(n.func.value.func, ast.Name)
        and n.func.value.func.id == "super"
    ]
    if len(super_calls) != 1 or ast.unparse(super_calls[0]) != "super().__init__(seed=seed, learning_rate=learning_rate)":
        raise ValueError("local initialization no longer delegates same seed and rate")
    # Descendant presence is insufficient: delegation must be an unconditional
    # direct statement on the normal initializer path.
    top_level_super = [
        statement for statement in local.body
        if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call)
        and ast.unparse(statement.value) == "super().__init__(seed=seed, learning_rate=learning_rate)"
    ]
    if len(top_level_super) != 1:
        raise ValueError("local base initialization is not on the top-level path")
    if "self.local_rate = float(local_rate)" not in local_text or "self.logical_counters = Counters()" not in local_text:
        raise ValueError("local-only state initialization changed")
    return {
        "kind": "CL6D_STATIC_INITIALIZER_SOURCE_AUDIT",
        "decision": "NO_GO",
        "sparse_and_shared_inherit_local_initializer": True,
        "local_delegates_seed_and_rate_to_base": True,
        "base_parameter_fields": sorted(EXPECTED_BASE_FIELDS),
        "local_only_fields": sorted(EXPECTED_LOCAL_FIELDS),
        "source_sha256": {key: hashlib.sha256(src.encode("utf-8")).hexdigest()
                          for key, src in sorted(sources.items())},
        "model_instantiated": False,
        "model_inference_executed": False,
        "training_executed": False,
        "dataset_generated": False,
        "initial_weights_compared": False,
        "numerical_parity_approved": False,
        "caveat": "AST source patterns only. Actual seeded weights, RNG state, numerical parity and training state NOT measured.",
    }


def audit() -> dict:
    parent = review(json.loads(CL6A_PATH.read_text(encoding="utf-8")))
    if parent["decision"] != "NO_GO" or parent["closed_gates"] != 0:
        raise ValueError("CL-6A gates changed")
    return audit_initializers({
        key: path.read_text(encoding="utf-8") for key, path in PATHS.items()
    })


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    result = audit()
    print("CL-6D — static initialization source audit (NO TRAINING)")
    print("Decision: NO_GO; no models constructed or executed.")
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
