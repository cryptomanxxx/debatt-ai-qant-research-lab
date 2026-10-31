"""No optional simulator packages are needed to run these checks."""

import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from integrations.continual_learning.check_backends import inspect_backends, main


class BackendAvailabilityTests(unittest.TestCase):
    def test_missing_source_is_not_mistaken_for_simphony_installation(self):
        with tempfile.TemporaryDirectory() as folder:
            result = inspect_backends(folder)
        self.assertFalse(result["simphony_source"]["available"])
        self.assertIn("not validated", result["simphony_source"]["evidence"])

    def test_source_signature_is_only_availability_evidence(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "onnarchsim" / "simulator.py"
            source.parent.mkdir()
            source.touch()
            result = inspect_backends(folder)
        self.assertTrue(result["simphony_source"]["available"])
        self.assertEqual("verified onnarchsim/simulator.py exists; runtime and CACTI not validated", result["simphony_source"]["evidence"])

    def test_probe_never_imports_simulator_or_runs_models(self):
        with patch("integrations.continual_learning.check_backends.importlib.util.find_spec", return_value=None) as finder:
            result = inspect_backends()
        self.assertFalse(result["lava_cpu"]["available"])
        self.assertFalse(result["qant_cpu"]["available"])
        self.assertEqual(2, finder.call_count)

    def test_no_strict_mode_passes_with_optional_tools_absent(self):
        with patch("integrations.continual_learning.check_backends.inspect_backends", return_value={
            "qant_cpu": {"available": False},
            "lava_cpu": {"available": False},
            "simphony_source": {"available": False},
        }), patch("sys.argv", ["check_backends.py"]):
            self.assertEqual(0, main())


if __name__ == "__main__":
    unittest.main()
