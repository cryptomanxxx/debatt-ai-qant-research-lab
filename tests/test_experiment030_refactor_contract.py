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
import subprocess
import sys
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
# ast.dump() is version dependent. Both digests are from the original
# verified blob, using each supported interpreter's AST representation.
LEGACY_AST_SHA256_BY_VERSION = {
    (3, 11): "46102e911ec43c0ed95c9950abb4f8ce3b4ed39975dcaa5e73182dcfa9bb6852",
    (3, 13): "870783abe27a066ea932deb0a7515aa31b4c412fad4c844ad0ec26383b04a05a",
}


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
        interpreter = sys.version_info[:2]
        self.assertIn(interpreter, LEGACY_AST_SHA256_BY_VERSION)
        self.assertEqual(
            hashlib.sha256(signature.encode()).hexdigest(),
            LEGACY_AST_SHA256_BY_VERSION[interpreter],
        )

        # For checkouts retaining the pre-refactor base tree, compare the
        # original Git blob's AST under this SAME Python interpreter too.
        # Future shallow checkouts might not contain that historical blob;
        # in that case the version-specific pinned digest still applies.
        original = subprocess.run(
            ["git", "cat-file", "blob", LEGACY_GIT_BLOB_SHA1],
            cwd=EXPERIMENT.parents[2], capture_output=True, check=False,
        )
        if original.returncode == 0:
            self.assertEqual(
                hashlib.sha1(
                    b"blob " + str(len(original.stdout)).encode()
                    + b"\\x00" + original.stdout
                ).hexdigest(),
                LEGACY_GIT_BLOB_SHA1,
            )
            old_tree = ast.parse(
                original.stdout.decode("utf-8"), filename="exp030_legacy.py"
            )
            self.assertEqual(
                signature, ast.dump(
                    old_tree, annotate_fields=True, include_attributes=False
                ),
            )
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
