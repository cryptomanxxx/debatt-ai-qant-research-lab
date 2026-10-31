"""Offline CL-3C synthetic data provenance checks; no training."""
import contextlib
import hashlib
import io
import json
import unittest

from integrations.continual_learning.cl3c_dataset_audit import (
    GENERATOR_PATH,
    audit_seed,
    main,
    prepare_audit,
)


class CL3CDatasetAuditContracts(unittest.TestCase):
    def test_paired_seed_audit_is_reproducible_and_disjoint(self):
        report = prepare_audit()
        self.assertFalse(report["training_performed"])
        self.assertFalse(report["experiment_authorized"])
        self.assertEqual(
            hashlib.sha256(GENERATOR_PATH.read_bytes()).hexdigest(),
            report["generator_source_sha256"],
        )
        self.assertEqual([101, 102, 103, 104, 105],
                         [x["seed"] for x in report["paired_seeds"]])
        self.assertEqual(5, len({x["dataset_content_sha256"] for x in report["paired_seeds"]}))
        for seed_report in report["paired_seeds"]:
            self.assertEqual(768, seed_report["unique_input_vectors"])
            self.assertEqual(4, len(seed_report["tasks"]))
            for task in seed_report["tasks"]:
                for name, expected in (("train", 128), ("test", 64)):
                    part = task[name]
                    self.assertEqual(expected, part["count"])
                    self.assertEqual(expected // 2, part["label_0"])
                    self.assertEqual(expected // 2, part["label_1"])
                    self.assertTrue(0 <= part["realized_activity_fraction"] <= 1)
        self.assertEqual(report["paired_seeds"][0], audit_seed(101, 128, 64))

    def test_cli_has_only_audit_mode_and_no_train_mode(self):
        with contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(0, main(["--audit"]))
        report = json.loads(out.getvalue())
        self.assertFalse(report["training_performed"])
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as caught:
                main(["--train"])
        self.assertEqual(2, caught.exception.code)


if __name__ == "__main__":
    unittest.main()
