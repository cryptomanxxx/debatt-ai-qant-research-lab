"""CL-13 guard tests: no dataset generator invoked."""
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from integrations.continual_learning.cl13_data_only import (
    CONFIRMATION, generate_verified_only,
)


class TestCL13Guard(unittest.TestCase):
    def test_requires_exact_explicit_confirmation(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch.dict(os.environ, {"CL13_DATA_ONLY": "1"}):
                with self.assertRaises(PermissionError):
                    generate_verified_only(Path(folder) / "not-written", "WRONG")

    def test_requires_separate_environment_gate(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch.dict(os.environ, {"CL13_DATA_ONLY": "0"}):
                with self.assertRaises(PermissionError):
                    generate_verified_only(Path(folder) / "not-written", CONFIRMATION)

    def test_never_overwrites_existing_output(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch.dict(os.environ, {"CL13_DATA_ONLY": "1"}):
                with self.assertRaises(FileExistsError):
                    generate_verified_only(Path(folder), CONFIRMATION)


if __name__ == "__main__":
    unittest.main()
