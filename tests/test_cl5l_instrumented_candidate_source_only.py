"""CL-5L source-only tests. NEVER invoke either dataset generator."""
import ast
from pathlib import Path
import unittest

from integrations.continual_learning import cl2_shared_rule_candidate

ROOT = Path(cl2_shared_rule_candidate.__file__).resolve().parent
ORIGINAL = ROOT / "cl2_shared_rule_candidate.py"
INSTRUMENTED = ROOT / "cl5l_instrumented_candidate.py"


def function_ast(path, name):
    module = ast.parse(path.read_text(encoding="utf-8"))
    matches = [n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == name]
    if len(matches) != 1:
        raise ValueError("expected exactly one target function")
    return matches[0]


class InstrumentationSourceOnlyTests(unittest.TestCase):
    def test_original_and_variant_keep_seed_recipe_and_one_draw_site(self):
        original = function_ast(ORIGINAL, "generate_shared_rule_candidate")
        variant = function_ast(INSTRUMENTED, "generate_shared_rule_candidate_with_counters")
        for tree in (original, variant):
            source = ast.unparse(tree)
            self.assertIn('{VERSION}|{seed}|{task_id}', source)
            self.assertIn("random.Random(int.from_bytes(material, 'big'))", source)
            self.assertIn("rng.random() < probability", source)
            calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                     and isinstance(n.func, ast.Attribute) and n.func.attr == "random"]
            self.assertEqual(len(calls), 1)
            self.assertIn("for _ in range(1000000):", source)

    def test_connected_report_provenance_is_explicit(self):
        source = ast.unparse(function_ast(
            INSTRUMENTED, "generate_shared_rule_candidate_with_counters"
        ))
        self.assertIn("report['generator_connected'] = True", source)
        self.assertIn("report['dataset_generation_executed'] = True", source)
        self.assertIn("CL5L_INSTRUMENTED_GENERATOR_PARTITION", source)

    def test_variant_module_has_no_top_level_generation_call(self):
        module = ast.parse(INSTRUMENTED.read_text(encoding="utf-8"))
        self.assertFalse(any(isinstance(n, (ast.Expr, ast.Assign, ast.AnnAssign))
                             and any(isinstance(c, ast.Call) and isinstance(c.func, ast.Name)
                                     and c.func.id == "generate_shared_rule_candidate_with_counters"
                                     for c in ast.walk(n))
                             for n in module.body))


if __name__ == "__main__":
    unittest.main()
