"""CL-10L source-only AST checks. No model objects or forward calls."""
import ast
import unittest
from pathlib import Path

from integrations.continual_learning.cl10l_forward_hooks import attached_candidate

HERE = Path(__file__).resolve().parent
SOURCES = (
    ("cl2_mlp_baseline.py", "TinyDenseMLP", "_forward", "dense"),
    ("cl2_sparse_forward.py", "SparseForwardLocalMLP", "_forward", "sparse"),
    ("cl2_shared_index_forward.py", "SharedIndexSparseForwardLocalMLP",
     "_forward_with_active", "shared"),
)


def method_ast(path, cls_name, name):
    tree = ast.parse((HERE / path).read_text(encoding="utf-8"))
    cls = next(n for n in tree.body
               if isinstance(n, ast.ClassDef) and n.name == cls_name)
    return next(n for n in cls.body
                if isinstance(n, ast.FunctionDef) and n.name == name)


class TestCL10L(unittest.TestCase):
    def test_dispatch_is_opt_in_and_default_source_remains(self):
        for path, cls, name, mode in SOURCES:
            with self.subTest(path=path):
                method = method_ast(path, cls, name)
                self.assertIsInstance(method.body[0], ast.If)
                self.assertEqual(ast.unparse(method.body[0].test),
                                 "attached_candidate(self)")
                dispatch = ast.unparse(method.body[0])
                self.assertIn(f"mode='{mode}'", dispatch)
                self.assertGreater(len(method.body), 1)
                # The reviewed original forward source remains present
                # after the optional branch, awaiting future numeric parity.
                self.assertTrue(any(
                    isinstance(n, ast.Call) and
                    isinstance(n.func, ast.Attribute) and
                    n.func.attr == "tanh" for n in ast.walk(method)
                ))

    def test_helper_has_no_hidden_rescan_for_shared_mode(self):
        src = (HERE / "cl10l_forward_hooks.py").read_text(encoding="utf-8")
        tree = ast.parse(src)
        method = next(n for n in tree.body
                      if isinstance(n, ast.FunctionDef)
                      and n.name == "candidate_forward")
        branch = next(n for n in ast.walk(method)
                      if isinstance(n, ast.If)
                      and ast.unparse(n.test) == "mode == 'shared'")
        self.assertNotIn("enumerate(x)", ast.unparse(branch))

    def test_unattached_fake_object_does_not_activate_forward(self):
        class Fake:
            pass
        self.assertFalse(attached_candidate(Fake()))


if __name__ == "__main__":
    unittest.main()
