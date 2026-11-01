"""CL-6B fail-closed source-level equivalence audit, with NO model imports.

Checks the current exact AST method graph and update targets for local arms.
No model construction, forward pass, prediction, data generation or training.
Source equality at reviewed AST sites is NOT proof of numerical equivalence,
equal initialization during real runs, or experimental causal validity.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
from pathlib import Path

from integrations.continual_learning.cl6a_readiness_review import PATH as PROTOCOL, review
import json

HERE = Path(__file__).resolve().parent
PATHS = {
    "dense": HERE / "cl2_mlp_baseline.py",
    "local": HERE / "cl2_local_mlp.py",
    "sparse": HERE / "cl2_sparse_forward.py",
    "shared": HERE / "cl2_shared_index_forward.py",
}
CLASSES = {
    "dense": ("TinyDenseMLP", []),
    "local": ("LocalHebbianMLP", ["TinyDenseMLP"]),
    "sparse": ("SparseForwardLocalMLP", ["LocalHebbianMLP"]),
    "shared": ("SharedIndexSparseForwardLocalMLP", ["SparseForwardLocalMLP"]),
}


def _class(source: str, name: str, bases: list[str]) -> ast.ClassDef:
    module = ast.parse(source)
    matches = [n for n in module.body if isinstance(n, ast.ClassDef) and n.name == name]
    if len(matches) != 1:
        raise ValueError(f"missing/duplicated model class {name}")
    found = [ast.unparse(base) for base in matches[0].bases]
    if found != bases:
        raise ValueError(f"{name}: inheritance changed, needs independent review")
    return matches[0]


def _methods(cls: ast.ClassDef) -> dict[str, ast.FunctionDef]:
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef)]
    result = {m.name: m for m in methods}
    if len(result) != len(methods):
        raise ValueError("duplicate methods")
    return result


def audit_sources(sources: dict[str, str]) -> dict:
    if set(sources) != set(PATHS) or any(type(s) is not str for s in sources.values()):
        raise ValueError("four exact source files required")
    models = {}
    methods = {}
    for key, (name, bases) in CLASSES.items():
        models[key] = _class(sources[key], name, bases)
        methods[key] = _methods(models[key])
    if "step" not in methods["local"]:
        raise ValueError("local update rule missing")
    if "step" in methods["sparse"] or "step" in methods["shared"]:
        raise ValueError("sparse variant overrides local update step")
    # Dynamic calls in inherited step (notably self._input and cost hooks)
    # must not be silently overridden by either ablation subclass.
    if set(methods["sparse"]) != {"_forward_first_layer_reads", "_forward"}:
        raise ValueError("sparse subclass method set changed; inherited dispatch unreviewed")
    if set(methods["shared"]) != {"_forward_with_active"}:
        raise ValueError("shared subclass method set changed; inherited dispatch unreviewed")
    # The inherited hook must dynamically dispatch to self._forward(x).
    # Calling TinyDenseMLP._forward(self, x) here silently bypasses sparse W1.
    hook = methods["local"].get("_forward_with_active")
    if hook is None:
        raise ValueError("local active-input forward hook missing")
    non_doc = [n for n in hook.body if not (
        isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant)
        and isinstance(n.value.value, str)
    )]
    if (len(non_doc) != 1 or not isinstance(non_doc[0], ast.Return)
            or non_doc[0].value is None
            or ast.unparse(non_doc[0].value) != "self._forward(x)"):
        raise ValueError("local hook must dynamically delegate to self._forward(x)")
    if "step" not in methods["dense"]:
        raise ValueError("dense reference update missing")
    if "_forward" not in methods["dense"] or "_forward" not in methods["sparse"]:
        raise ValueError("forward reference missing")
    if "_forward_with_active" not in methods["local"] or "_forward_with_active" not in methods["shared"]:
        raise ValueError("active-index hook missing")
    if "_forward_with_active" in methods["sparse"] or "_forward" in methods["shared"]:
        raise ValueError("unreviewed sparse/shared forward dispatch override")
    local_step = methods["local"]["step"]
    calls = [ast.unparse(n.func) for n in ast.walk(local_step) if isinstance(n, ast.Call)]
    if calls.count("self._forward_with_active") != 1:
        raise ValueError("local step no longer uses shared forward hook")
    if calls.count("self._input") != 1:
        raise ValueError("local step input validation changed")
    updates = [ast.unparse(n.target) for n in ast.walk(local_step)
               if isinstance(n, ast.AugAssign)]
    expected = {"self.w1[j][i]", "self.b1[j]", "self.w2[k][j]", "self.b2[k]"}
    bookkeeping = {"self.logical_counters", "self.optimizer_steps"}
    if set(updates) != expected | bookkeeping or len(updates) != 6:
        raise ValueError("local weight update targets changed")
    def _self_state(target: ast.AST) -> bool:
        while isinstance(target, (ast.Subscript, ast.Attribute)):
            target = target.value
        return isinstance(target, ast.Name) and target.id == "self"

    def _rooted_at_name(target: ast.AST, name: str) -> bool:
        while isinstance(target, (ast.Subscript, ast.Attribute)):
            target = target.value
        return isinstance(target, ast.Name) and target.id == name

    # Reject ordinary assignments to self.w1[0][0], as well as augmented
    # writes, annotated writes, deletions, and self-state mutation calls.
    # Assignments to local variables such as hidden and logits remain allowed.
    for key in ("dense", "local", "sparse", "shared"):
        for method in methods[key].values():
            if not method.name.startswith("_forward"):
                continue
            for node in ast.walk(method):
                targets = ()
                if isinstance(node, ast.Assign):
                    targets = node.targets
                elif isinstance(node, (ast.AugAssign, ast.AnnAssign, ast.NamedExpr)):
                    targets = (node.target,)
                elif isinstance(node, ast.Delete):
                    targets = node.targets
                if any(_self_state(t) for t in targets):
                    raise ValueError("forward method writes to model state")
                # `active` is caller-owned by LocalHebbianMLP.step and reused
                # for W1 updates. The shared hook must not alter it.
                if key == "shared" and method.name == "_forward_with_active":
                    if any(_rooted_at_name(t, "active") for t in targets):
                        raise ValueError("shared forward mutates caller-owned active indices")
                    if (isinstance(node, ast.Call)
                            and isinstance(node.func, ast.Attribute)
                            and _rooted_at_name(node.func.value, "active")):
                        raise ValueError("shared forward mutates caller-owned active indices")
                if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                        and _self_state(node.func.value)
                        and not (key == "local" and method.name == "_forward_with_active"
                                 and ast.unparse(node.func) == "self._forward")):
                    raise ValueError("forward method calls unreviewed self-state operation")
    # This checks the narrow AST contract. It does not certify numerical equality.
    dense_first = ast.unparse(methods["dense"]["_forward"])
    sparse_first = ast.unparse(methods["sparse"]["_forward"])
    shared_first = ast.unparse(methods["shared"]["_forward_with_active"])
    if ("range(INPUT_SIZE)" not in dense_first
            or "for i in active" not in sparse_first
            or "for i in active" not in shared_first):
        raise ValueError("dense/sparse traversal distinction changed")
    for current in (dense_first, sparse_first, shared_first):
        if "math.tanh" not in current or "self.w2" not in current:
            raise ValueError("nonlinearity or output forward missing")
    return {
        "kind": "CL6B_STATIC_MODEL_SOURCE_EQUIVALENCE_AUDIT",
        "decision": "NO_GO",
        "local_step_inherited_by_sparse": True,
        "local_step_inherited_by_shared": True,
        "shared_index_changes_forward_hook_only": True,
        "weight_update_targets": sorted(expected),
        "source_sha256": {
            key: hashlib.sha256(src.encode("utf-8")).hexdigest()
            for key, src in sorted(sources.items())
        },
        "model_instantiated": False,
        "forward_or_inference_executed": False,
        "training_executed": False,
        "dataset_generated": False,
        "physical_energy_measured": False,
        "scientific_equivalence_approved": False,
        "caveat": "AST-level inheritance/update-site contract only; numerical equivalence, "
                  "seed/order parity, state comparison and actual costs remain unverified",
    }


def audit() -> dict:
    # Reuse CL-6A scientific NO_GO; do not mark any of its gates resolved.
    result = review(json.loads(PROTOCOL.read_text(encoding="utf-8")))
    if result["decision"] != "NO_GO" or result["closed_gates"] != 0:
        raise ValueError("CL-6A scientific status changed")
    return audit_sources({
        key: path.read_text(encoding="utf-8") for key, path in PATHS.items()
    })


def render_summary(result: dict) -> str:
    lines = [
        "## CL-6B — source-level arm equivalence audit (NO TRAINING)",
        "",
        "> **NO_GO**: AST patterns and method inheritance only. No model was "
        "constructed or executed; zero learning experiments or measurements.",
        "",
        "| Source evidence | Static finding |",
        "|---|---|",
        "| Sparse local | Inherits `LocalHebbianMLP.step` unchanged |",
        "| Shared-index local | Inherits the same `step` unchanged |",
        "| Sparse traversal | Overrides `_forward` for active W1 indices |",
        "| Shared-index traversal | Overrides `_forward_with_active` for reuse inside step |",
        "| Common supervised + Hebbian-like updates | Four update targets remain in inherited step |",
        "",
        "### Source fingerprints (SHA-256 of UTF-8 source)",
        "",
    ]
    for name, digest in result["source_sha256"].items():
        lines.append(f"- `{name}`: `{digest}`")
    lines += [
        "",
        "**Not established:** identical floating-point outputs, exact initialized "
        "weight equality on paired seeds, fixed sample order, matched learning "
        "trajectories, cost-counter parity, hardware memory traffic or real energy.",
        "",
        "**All nine CL-6A review/authorization gates remain OPEN.** "
        "Independent scientific review and Marcus's separate explicit approval "
        "are required before experimental training.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args(argv)
    result = audit()
    text = render_summary(result)
    if args.summary is not None:
        with args.summary.open("a", encoding="utf-8") as stream:
            stream.write(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
