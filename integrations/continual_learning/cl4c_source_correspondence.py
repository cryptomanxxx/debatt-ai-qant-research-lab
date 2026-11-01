"""CL-4C fail-closed static AST correspondence audit; never executes model code."""
from __future__ import annotations

import ast
from pathlib import Path

from integrations.continual_learning.cl3y_readiness_audit import scientific_readiness_audit
from integrations.continual_learning.cl4b_source_event_probe import source_loop_probe

HERE = Path(__file__).resolve().parent
DENSE_PATH = HERE / "cl2_mlp_baseline.py"
SPARSE_PATH = HERE / "cl2_sparse_forward.py"


def _read_source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _forward_method(path: Path, class_name: str) -> ast.FunctionDef:
    module = ast.parse(_read_source(path))
    classes = [n for n in module.body if isinstance(n, ast.ClassDef) and n.name == class_name]
    if len(classes) != 1:
        raise ValueError(f"expected exactly one {class_name}")
    forwards = [n for n in classes[0].body if isinstance(n, ast.FunctionDef) and n.name == "_forward"]
    if len(forwards) != 1:
        raise ValueError("forward source method missing or ambiguous")
    return forwards[0]


def _sum_generators(method: ast.FunctionDef) -> list[str]:
    sums = [n for n in ast.walk(method)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "sum"]
    if len(sums) != 2 or any(len(n.args) != 1 or not isinstance(n.args[0], ast.GeneratorExp) for n in sums):
        raise ValueError("forward traversal changed; review required")
    return [ast.unparse(n.args[0]) for n in sums]


def _calls(method: ast.FunctionDef, function: str) -> int:
    return sum(isinstance(n, ast.Call) and ast.unparse(n.func) == function for n in ast.walk(method))


def _check_enclosing_layer_bounds(method: ast.FunctionDef) -> None:
    """Check comprehensions surrounding the two sum generators, not only sums."""
    comps = [n for n in ast.walk(method) if isinstance(n, ast.ListComp)]
    hidden = [n for n in comps if any(
        isinstance(call, ast.Call) and ast.unparse(call.func) == "math.tanh"
        for call in ast.walk(n.elt)
    )]
    output = [n for n in comps if isinstance(n.elt, ast.BinOp)
              and ast.unparse(n.elt.left) == "self.b2[k]"]
    for matches, target, iteration, label in (
        (hidden, "j", "range(HIDDEN_SIZE)", "hidden"),
        (output, "k", "range(N_CLASSES)", "output"),
    ):
        if len(matches) != 1 or len(matches[0].generators) != 1:
            raise ValueError(f"{label} enclosing traversal missing or ambiguous")
        gen = matches[0].generators[0]
        if ast.unparse(gen.target) != target or ast.unparse(gen.iter) != iteration or gen.ifs or gen.is_async:
            raise ValueError(f"{label} enclosing traversal bounds changed")


def source_correspondence_audit() -> dict:
    """Check the two currently reviewed CL-2 first-layer source patterns."""
    dense = _forward_method(DENSE_PATH, "TinyDenseMLP")
    sparse = _forward_method(SPARSE_PATH, "SparseForwardLocalMLP")
    dense_sums = _sum_generators(dense)
    sparse_sums = _sum_generators(sparse)
    _check_enclosing_layer_bounds(dense)
    _check_enclosing_layer_bounds(sparse)
    dense_first = "(self.w1[j][i] * x[i] for i in range(INPUT_SIZE))"
    sparse_first = "(self.w1[j][i] for i in active)"
    second = "(self.w2[k][j] * hidden[j] for j in range(HIDDEN_SIZE))"
    # ast.walk is breadth-first: nesting can change the visitation order.
    # Classify by exact generator identity, not the order in which sum() is visited.
    if set(dense_sums) != {dense_first, second}:
        raise ValueError("dense first-layer/output traversal drift")
    if set(sparse_sums) != {sparse_first, second}:
        raise ValueError("sparse first-layer/output traversal drift")
    if _calls(dense, "math.tanh") != 1 or _calls(sparse, "math.tanh") != 1:
        raise ValueError("hidden nonlinearity site changed")
    sparse_lists = [n for n in ast.walk(sparse) if isinstance(n, ast.ListComp)
                    and any(isinstance(child, ast.Call) and ast.unparse(child.func) == "enumerate"
                            for child in ast.walk(n))]
    if len(sparse_lists) != 1 or ast.unparse(sparse_lists[0]) != "[i for i, value in enumerate(x) if value]":
        raise ValueError("sparse active-index source changed")
    # Distinct from executing a neural model: validate loop-probe counts only.
    example = (1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0)
    counts_dense = source_loop_probe(example, sparse=False)["counters"]
    counts_sparse = source_loop_probe(example, sparse=True)["counters"]
    if counts_dense["forward_weight_terms"] != 144 or counts_sparse["forward_weight_terms"] != 32:
        raise ValueError("source-loop probe contract changed")
    audit = scientific_readiness_audit()
    if audit["decision"] != "NO_GO" or audit["open_count"] != 9:
        raise ValueError("CL-3Y scientific gates changed")
    return {
        "kind": "CL4C_STATIC_SOURCE_CORRESPONDENCE_ONLY",
        "dense_first_layer": dense_first,
        "sparse_first_layer": sparse_first,
        "shared_output_layer": second,
        "gates_open": 9,
        "decision": "NO_GO",
        "models_executed": False,
        "training_allowed": False,
        "energy_measured": False,
        "assurance_scope": "specific AST traversal patterns only; not numerical equivalence",
    }
