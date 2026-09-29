"""Offline AST equivalence contract for the format-only Experiment 030 refactor.

Baseline: experiments/030_qant_compatibility_model_v3/run.py as it existed on
main before PR #152, Git blob 64b1465edbaf65b57a33ab67a083cab522bba9d4.
Original parsed using Python 3.11 and ast.dump(annotate_fields=True,
include_attributes=False). This test NEVER imports/executes run.py: doing so
would initiate a real training job and dataset access in the legacy script.
"""
import ast
import hashlib
import io
import tokenize
import unittest
from pathlib import Path

EXPERIMENT = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "030_qant_compatibility_model_v3"
    / "run.py"
)
LEGACY_GIT_BLOB_SHA1 = "64b1465edbaf65b57a33ab67a083cab522bba9d4"
LEGACY_AST_SHA256 = (
    "870783abe27a066ea932deb0a7515aa31b4c412fad4c844ad0ec26383b04a05a"
)


class Experiment030FormattingContract(unittest.TestCase):
    def test_exact_python_311_ast_is_identical_to_original(self):
        # AST comparison is much stronger than grepping individual operations:
        # all imports, statements, function bodies, call argument ordering,
        # names, literals, metric formulas and output schema are included.
        source = EXPERIMENT.read_text(encoding="utf-8")
        syntax_tree = ast.parse(source, filename=str(EXPERIMENT))
        signature = ast.dump(
            syntax_tree, annotate_fields=True, include_attributes=False
        )
        self.assertEqual(hashlib.sha256(signature.encode()).hexdigest(),
                         LEGACY_AST_SHA256)
        # Syntax-to-bytecode compilation does NOT execute imports, dataset
        # downloads, Q.ANT toolkit operations, training or result writes.
        self.assertIsNotNone(compile(syntax_tree, str(EXPERIMENT), "exec"))

    def test_readability_without_runtime_changes(self):
        source = EXPERIMENT.read_text(encoding="utf-8")
        self.assertGreater(len(source.splitlines()), 150)
        self.assertTrue(all(len(line) <= 100 for line in source.splitlines()))
        operators = [
            tok.string for tok in tokenize.generate_tokens(io.StringIO(source).readline)
            if tok.type == tokenize.OP
        ]
        self.assertNotIn(";", operators, "do not reintroduce semicolon chains")


if __name__ == "__main__":
    unittest.main()
