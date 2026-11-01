"""CL-8D fixed non-model Linux isolation probe regression tests."""
from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from integrations.continual_learning import cl8d_linux_isolation_probe as probe


class CL8DIsolationProbeTests(unittest.TestCase):
    def test_pure_positive_proof_contract(self):
        valid = json.dumps({
            "probe": "CL8D_NO_MODEL", "net_namespace": "net:[90002]"
        }).encode("utf-8")
        result = probe.validate_probe_response(0, valid, b"", "net:[90001]")
        self.assertEqual(result["status"], "ISOLATED_PROBE_OK")
        self.assertFalse(result["general_model_runner_available"])
        self.assertFalse(result["training_executed"])

    def test_rejects_absent_or_unchanged_network_namespace(self):
        for value in (
            {"probe": "CL8D_NO_MODEL", "net_namespace": "net:[90001]"},
            {"probe": "CL8D_NO_MODEL", "net_namespace": "unknown"},
            {"probe": "CL8D_NO_MODEL", "net_namespace": "net:[90002]",
             "experiment_approved": True},
            {"probe": "CL8D_TRAINING", "net_namespace": "net:[90002]"},
        ):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    probe.validate_probe_response(
                        0, json.dumps(value).encode("utf-8"), b"", "net:[90001]"
                    )

    def test_rejects_failed_unshare_and_oversized_output(self):
        with self.assertRaises(ValueError):
            probe.validate_probe_response(
                1, b"", b"unshare: operation not permitted", "net:[9]"
            )
        with self.assertRaises(ValueError):
            probe.validate_probe_response(
                0, b"x" * (probe.OUTPUT_LIMIT_BYTES + 1), b"", "net:[9]"
            )

    def test_failed_child_reports_sanitized_failure_category(self):
        samples = (
            (b"unshare: unshare failed: Operation not permitted", "OS_PERMISSION_DENIED"),
            (b"unshare: Permission denied", "OS_PERMISSION_DENIED"),
            (b"unshare: unrecognized option '--net'", "UNSHARE_OPTION_UNSUPPORTED"),
            (b"malicious \`example\` and arbitrary output", "CHILD_FAILURE_UNCLASSIFIED"),
        )
        for stderr, expected in samples:
            with self.subTest(expected=expected):
                with self.assertRaisesRegex(ValueError, expected) as raised:
                    probe.validate_probe_response(1, b"", stderr, "net:[9]")
                self.assertNotIn("malicious", str(raised.exception))

    def test_missing_unshare_never_launches_child(self):
        with patch.object(probe.platform, "system", return_value="Linux"), \
             patch.object(probe.Path, "exists", return_value=True), \
             patch.object(probe, "_trusted_unshare", return_value=None), \
             patch.object(probe.subprocess, "Popen") as popen:
            with self.assertRaises(RuntimeError):
                probe.probe()
            popen.assert_not_called()

    def test_caller_path_cannot_select_writable_wrapper(self):
        with patch.object(probe.Path, "resolve", return_value=probe.Path("/tmp/writable/unshare")):
            self.assertIsNone(probe._trusted_unshare())

    def test_rejects_unapproved_command_line_modes(self):
        with self.assertRaises(SystemExit):
            probe.main(["--execute"])


if __name__ == "__main__":
    unittest.main()
