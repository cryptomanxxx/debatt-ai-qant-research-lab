"""CL-18 structural source parity: parse Python AST only, NEVER import models.

This is a source-inheritance and update-path preflight, NOT numeric parity.
Actual float trace comparison requires separate model-inference authorization.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
FILES = {
    "dense": ("cl2_mlp_baseline.py", "TinyDenseMLP", None),
    "local": ("cl2_local_mlp.py", "LocalHebbianMLP", "TinyDenseMLP"),
    "sparse": ("cl2_sparse_forward.py", "SparseForwardLocalMLP", "LocalHebbianMLP"),
    "shared": ("cl2_shared_index_forward.py", "SharedIndexSparseForwardLocalMLP", "SparseForwardLocalMLP"),
}


def _fail(reason: str) -> None:
    raise ValueError("CL-18 NO_GO: " + reason)


def audit_source_texts(sources: dict[str, str]) -> dict:
    """Verify expected inheritance/update boundaries on supplied source text."""
    if type(sources) is not dict or set(sources) != set(FILES):
        _fail("source mapping mismatch")
    methods = {}
    for arm, (_, class_name, expected_parent) in FILES.items():
        tree = ast.parse(sources[arm])
        classes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name]
        if len(classes) != 1:
            _fail("missing or duplicate model class " + arm)
        cls = classes[0]
        # Every base must be checked. Ignoring qualified/mixed-in bases can
        # silently alter MRO and replace the inherited update implementation.
        if any(not isinstance(base, ast.Name) for base in cls.bases):
            _fail("unsupported qualified or dynamic inheritance " + arm)
        parents = [base.id for base in cls.bases]
        if expected_parent is None:
            if parents:
                _fail("dense baseline unexpectedly inherited")
        elif parents != [expected_parent]:
            _fail("inheritance drift " + arm)
        if cls.decorator_list or cls.keywords:
            _fail("decorated or parameterized class inheritance " + arm)
        # Current model classes contain only methods and docstrings. Reject
        # class-level assignments (including lambda aliasing), exec(), and
        # other executable bodies that could replace step/predict/_forward.
        if any(not (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                    or (isinstance(node, ast.Expr)
                        and isinstance(node.value, ast.Constant)
                        and type(node.value.value) is str))
               for node in cls.body):
            _fail("class-level binding or execution drift " + arm)
        names = {n.name for n in cls.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
        methods[arm] = names
        if arm in ("sparse", "shared") and "step" in names:
            _fail("sparse ablation changed local update implementation")
        if arm == "shared" and "_forward" in names:
            _fail("shared index ablation changed standalone forward")
        if "predict" in names and arm != "dense":
            _fail("ablation changed baseline prediction method")
        if any(isinstance(n, ast.AsyncFunctionDef) for n in cls.body):
            _fail("unexpected async model method")
    if not {"__init__", "_forward", "step", "predict", "loss_and_gradients"} <= methods["dense"]:
        _fail("dense reference method drift")
    if not {"__init__", "_forward_with_active", "step"} <= methods["local"]:
        _fail("local update path drift")
    if not {"_forward", "_forward_first_layer_reads"} <= methods["sparse"]:
        _fail("sparse forward override drift")
    if "_forward_with_active" not in methods["shared"]:
        _fail("shared active index override missing")

    local_tree = ast.parse(sources["local"])
    local = next(n for n in local_tree.body
                 if isinstance(n, ast.ClassDef) and n.name == "LocalHebbianMLP")
    step = next(n for n in local.body if isinstance(n, ast.FunctionDef) and n.name == "step")
    calls = {n.attr for n in ast.walk(step) if isinstance(n, ast.Attribute)}
    if not {"_forward_with_active", "optimizer_steps", "w1", "w2", "b1", "b2"} <= calls:
        _fail("local update or forward callsite drift")
    if not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
               and n.func.id == "note_weight_update" for n in ast.walk(step)):
        _fail("source-update hook missing")

    return {
        "decision": "NO_GO",
        "scope": "STATIC_AST_SOURCE_INHERITANCE_ONLY",
        "verified_source_arcs": [
            "TinyDenseMLP -> LocalHebbianMLP",
            "LocalHebbianMLP -> SparseForwardLocalMLP",
            "SparseForwardLocalMLP -> SharedIndexSparseForwardLocalMLP",
        ],
        "sparse_arms_inherit_local_step": True,
        "dense_and_local_updates_intentionally_differ": True,
        "floating_point_equivalence_verified": False,
        "runtime_imports_performed": False,
        "inference_authorized": False,
        "training_authorized": False,
        "numerical_tolerance_independently_approved": False,
        "physical_energy": "NOT_MEASURED",
    }


def audit_current_sources() -> dict:
    sources = {arm: (HERE / filename).read_text(encoding="utf-8")
               for arm, (filename, _, _) in FILES.items()}
    return audit_source_texts(sources)


if __name__ == "__main__":
    print(json.dumps(audit_current_sources(), sort_keys=True))
