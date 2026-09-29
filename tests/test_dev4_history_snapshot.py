"""Snapshot integrity and seed leakage tests; no Q.ANT compute."""
import hashlib
import json
import subprocess
import sys
import shutil
import tempfile
import unittest
from pathlib import Path
from scripts.build_dev4_history_snapshot import build_snapshot, canonical, verify_snapshot


class HistoryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.protocol = {"dataset": {"name": "ECG200", "train_shape": [100, 96],
                                     "test_shape": [100, 96]},
                         "evaluation": {"seeds": [301, 302, 303, 304, 305]}}

    def record(self, seeds):
        return {"dataset": self.protocol["dataset"],
                "configuration": {"seeds": seeds, "epochs": 100, "learning_rate": 0.001,
                                  "batch_size": 32, "frequencies": [1, 2]},
                "summary": {"alpha5_w9": {"aggregate_qant_correct": 80 * len(seeds),
                                          "parameter_count": 5000},
                            "alpha5_w16_control": {"aggregate_qant_correct": 82 * len(seeds),
                                                   "parameter_count": 8550}}}

    def test_snapshot_deterministic_and_rejects_seed_overlap(self):
        a, b = self.root / "a.json", self.root / "b.json"
        a.write_text(json.dumps(self.record([201, 202, 203, 204, 205])))
        b.write_text(json.dumps(self.record([301, 302, 303, 304, 305])))
        files = [a, b]
        snap = build_snapshot(files, self.protocol)
        self.assertEqual(len(snap["included"]), 1)
        self.assertEqual(snap["excluded"][0]["reason"], "evaluation_seed_overlap")
        digest = verify_snapshot(snap, files, self.protocol)
        self.assertEqual(digest, hashlib.sha256(canonical(snap)).hexdigest())
        self.assertEqual(canonical(snap), canonical(build_snapshot(list(reversed(files)), self.protocol)))
        a.write_text(a.read_text() + " ")
        with self.assertRaisesRegex(ValueError, "differs"):
            verify_snapshot(snap, files, self.protocol)

    def test_absolute_checkout_paths_produce_identical_snapshot(self):
        left = self.root / "checkout_one" / "results"
        right = self.root / "checkout_two" / "results"
        left.mkdir(parents=True)
        right.mkdir(parents=True)
        (left / "a.json").write_text(json.dumps(self.record([201, 202, 203, 204, 205])))
        shutil.copy2(left / "a.json", right / "a.json")
        one = build_snapshot([left / "a.json"], self.protocol, source_root=left)
        two = build_snapshot([right / "a.json"], self.protocol, source_root=right)
        self.assertEqual(canonical(one), canonical(two))
        self.assertEqual(one["included"][0]["source"], "a.json")
        self.assertEqual(verify_snapshot(one, [right / "a.json"], self.protocol, source_root=right),
                         hashlib.sha256(canonical(one)).hexdigest())

    def test_cli_output_inside_results_does_not_self_include(self):
        results = self.root / "results"
        results.mkdir()
        (results / "a.json").write_text(json.dumps(self.record([201, 202, 203, 204, 205])))
        output = results / "snapshot.json"
        protocol_file = self.root / "protocol.json"
        protocol_file.write_text(json.dumps(self.protocol))
        command = [sys.executable, "-m", "scripts.build_dev4_history_snapshot",
                   "--results", str(results), "--protocol", str(protocol_file),
                   "--output", str(output)]
        repo_root = Path(__file__).resolve().parents[1]
        first = subprocess.run(command, cwd=repo_root, capture_output=True, text=True, check=True)
        original = output.read_bytes()
        verified = subprocess.run(command + ["--verify"], cwd=repo_root,
                                  capture_output=True, text=True, check=True)
        self.assertIn(first.stdout.splitlines()[0], verified.stdout)
        subprocess.run(command, cwd=repo_root, capture_output=True, text=True, check=True)
        self.assertEqual(output.read_bytes(), original)
        self.assertEqual(len(json.loads(original)["excluded"]), 0)

    def test_incomplete_schema_excluded(self):
        a = self.root / "a.json"
        a.write_text(json.dumps({"dataset": self.protocol["dataset"]}))
        snap = build_snapshot([a], self.protocol)
        self.assertEqual(snap["included"], [])
        self.assertEqual(snap["excluded"][0]["reason"], "invalid_or_nonpaired_schema")


if __name__ == "__main__":
    unittest.main()
