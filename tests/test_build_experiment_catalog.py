"""Offline catalog contract tests: no SDK, network, Groq, or training imports."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.build_experiment_catalog import (
    CatalogError,
    ROOT,
    build_catalog,
    canonical_bytes,
)


def put(root, relative, data):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    return path


def fixture(root):
    toolkit_proposal = {
        "schema_version": 1, "proposal_id": "proposal-001-demo",
        "status": "proposed",
    }
    pnn_proposal = {
        "proposal_id": "pnn-v1-proposal001",
        "id": "PNN-v1-Proposal001",
        "experiment_id": "PNN-v1-Exp001", "status": "proposed",
    }
    put(root, "research_queue/proposals/proposal-001-demo.json", toolkit_proposal)
    put(root, "pnn-v1/proposals/proposal001.json", pnn_proposal)
    toolkit = {
        "schema_version": 1, "experiment_id": "exp006_demo",
        "backend": "qant-cpu", "dataset": "MNIST",
        "source_proposal": "proposal-001-demo",
        "job_id": "exp006-demo",
        "seeds": [11, 22],
        "configuration": {"epochs": 3},
        "summary": {"candidate": {"mean_qant_accuracy": 0.7}},
    }
    pnn = {
        "schema_version": 1, "experiment_id": "PNN-v1-Exp001",
        "proposal_id": "PNN-v1-Proposal001",
        "backend": "qant-cpu/software-simulation",
        "dataset": {"name": "ECG200", "split": "canonical TRAIN/TEST"},
        "configuration": {"seeds": [11, 22], "epochs": 100},
        "summary": {"alpha": {"aggregate_qant_correct": 180}},
    }
    put(root, "results/exp006_demo.json", toolkit)
    primary = put(root, "pnn-v1/results/result001.json", pnn)
    mirror = root / "results/result001.json"
    mirror.write_bytes(primary.read_bytes())
    put(root, "results/result_schema.example.json", {"experiment_id": "example"})
    put(root, "results/automation_smoke_test.json", {"test_id": "smoke"})
    put(root, "research_queue/jobs/exp006-demo.json", {
        "job_id": "exp006-demo", "status": "approved",
        "source_proposal": "proposal-001-demo",
    })
    put(root, "research_queue/jobs/pnn-v1-exp001.json", {
        "job_id": "pnn-v1-exp001", "status": "approved",
        "source_proposal": "pnn-v1-proposal001",
    })
    put(root, "executions/exp006-demo-123.json", {
        "job_id": "exp006-demo", "status": "completed",
        "git_commit": "a" * 40, "run_id": "123", "runner": "github-actions",
    })
    put(root, "executions/pnn-v1-exp001-456.json", {
        "job_id": "pnn-v1-exp001", "status": "completed",
        "git_commit": "b" * 40, "run_id": "456", "runner": "github-actions",
    })
    put(root, "research_queue/v3_hypotheses/registry.json", {
        "schema_version": 1, "record_kind": "unexecuted_hypothesis_registry",
        "training_runs": 0,
        "entries": [{
            "hypothesis_id": "v3-demo", "state": "proposed_untested",
            "source": {
                "workflow_run_id": 999, "original_report_sha256": "c" * 64,
            },
            "secret_holdout_result": "NEVER_INDEX_PROSPECTIVE_DATA",
        }],
    })
    put(root, "research_queue/v3_protocols/first_standalone_draft.json", {
        "schema_version": 1, "record_kind": "standalone_v3_protocol_draft",
        "protocol_id": "v3-draft-demo", "status": "draft_unapproved_unexecuted",
        "dataset": "ECG200",
        "execution": {"status": "not_authorized", "training_runs": 0,
                      "approval_record": None},
        "prospective_result": "NEVER_INDEX_PROSPECTIVE_DATA",
    })
    put(root, "research_queue/benchmarks/dev4_round2_verified_feedback.json", {
        "secret_strategy_feedback": "NEVER_INDEX_STRATEGY_FEEDBACK",
    })
    return primary, mirror


class GlobalCatalogTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.primary, self.mirror = fixture(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def test_inventory_is_stable_and_does_not_double_count_mirror(self):
        first = build_catalog(self.root)
        second = build_catalog(self.root)
        self.assertEqual(canonical_bytes(first), canonical_bytes(second))
        self.assertEqual(first["summary"]["canonical_completed_result_records"], 2)
        self.assertEqual(first["summary"]["exact_pnn_mirror_files"], 1)
        self.assertEqual(first["summary"]["records_by_kind"],
                         {"hypothesis": 1, "proposal": 2, "protocol_draft": 1, "result": 2})
        expected_ids = {
            "toolkit:result:exp006_demo:v1",
            "pnn-v1:result:PNN-v1-Exp001:v1",
            "pnn-v1:proposal:pnn-v1-proposal001:v1",
            "toolkit:proposal:proposal-001-demo:v1",
            "v3:hypothesis:v3-demo:v1",
            "v3:protocol_draft:v3-draft-demo:v1",
        }
        self.assertEqual({record["catalog_id"] for record in first["records"]},
                         expected_ids)
        self.assertEqual(first["records"], sorted(
            first["records"], key=lambda item: item["catalog_id"]))
        pnn = next(item for item in first["records"]
                   if item["catalog_id"] == "pnn-v1:result:PNN-v1-Exp001:v1")
        toolkit = next(item for item in first["records"]
                       if item["catalog_id"] == "toolkit:result:exp006_demo:v1")
        self.assertEqual(len(pnn["mirrored_sources"]), 1)
        self.assertEqual(pnn["source"]["sha256"], pnn["mirrored_sources"][0]["sha256"])
        self.assertEqual(len(pnn["execution_records"]), 1)
        self.assertEqual(pnn["execution_records"][0]["run_id"], "456")
        self.assertEqual(pnn["job_link_basis"], "numbered_experiment_job_convention_only")
        self.assertEqual(toolkit["job_link_basis"], "explicit_result_job_id")
        self.assertEqual(toolkit["execution_records"][0]["run_id"], "123")
        self.assertEqual(pnn["proposal_source"]["path"], "pnn-v1/proposals/proposal001.json")
        self.assertEqual(pnn["status"], "completed_recorded")
        self.assertFalse(pnn["result_artifact_independently_authenticated"])
        self.assertEqual(first["summary"]["linked_job_ids"], 2)

    def test_global_index_does_not_copy_holdout_or_strategy_feedback(self):
        content = canonical_bytes(build_catalog(self.root)).decode("utf-8")
        self.assertNotIn("NEVER_INDEX_STRATEGY_FEEDBACK", content)
        self.assertNotIn("NEVER_INDEX_PROSPECTIVE_DATA", content)
        self.assertIn("Do not feed this global index", content)
        self.assertNotIn("research_queue/benchmarks/dev4_round2_verified_feedback.json", content)
        for record in build_catalog(self.root)["records"]:
            if record["record_kind"] in ("hypothesis", "protocol_draft"):
                self.assertIsNone(record["result"])

    def test_blobs_and_manifest_change_with_source_bytes(self):
        one = build_catalog(self.root)
        pnn = next(item for item in one["records"] if item["series"] == "pnn-v1"
                   and item["record_kind"] == "result")
        raw = self.primary.read_bytes()
        expected_git_sha = hashlib.sha1(
            ("blob " + str(len(raw)) + "\0").encode() + raw).hexdigest()
        self.assertEqual(pnn["source"]["git_blob_sha1"], expected_git_sha)
        changed = json.loads(self.primary.read_text())
        changed["notes"] = "different primary bytes"
        put(self.root, "pnn-v1/results/result001.json", changed)
        # A divergent mirror must fail closed, not create another experiment.
        with self.assertRaisesRegex(CatalogError, "divergent PNN root mirror"):
            build_catalog(self.root)
        self.mirror.write_bytes(self.primary.read_bytes())
        two = build_catalog(self.root)
        self.assertNotEqual(one["source_manifest_sha256"], two["source_manifest_sha256"])

    def test_orphan_and_divergent_mirrors_fail_closed(self):
        self.mirror.unlink()
        put(self.root, "results/result002.json", {
            "experiment_id": "PNN-v1-Exp002", "backend": "qant-cpu",
        })
        with self.assertRaisesRegex(CatalogError, "orphan root PNN mirror"):
            build_catalog(self.root)

    def test_duplicate_identity_fails_even_without_root_mirror(self):
        self.mirror.unlink()
        another = json.loads(self.primary.read_text())
        put(self.root, "pnn-v1/results/result_extra.json", another)
        with self.assertRaisesRegex(CatalogError, "duplicate canonical PNN identity"):
            build_catalog(self.root)

    def test_numbered_filename_mismatch_fails(self):
        changed = json.loads(self.primary.read_text())
        changed["experiment_id"] = "PNN-v1-Exp002"
        put(self.root, "pnn-v1/results/result001.json", changed)
        with self.assertRaisesRegex(CatalogError, "PNN numbered filename/experiment_id mismatch"):
            build_catalog(self.root)

    def test_bad_job_and_execution_linkage_fails(self):
        path = self.root / "executions/pnn-v1-exp001-456.json"
        row = json.loads(path.read_text())
        row["run_id"] = "789"
        put(self.root, "executions/pnn-v1-exp001-456.json", row)
        with self.assertRaisesRegex(CatalogError, "execution filename/id mismatch"):
            build_catalog(self.root)

    def test_noncompleted_execution_cannot_attest_completed_result(self):
        path = self.root / "executions/pnn-v1-exp001-456.json"
        row = json.loads(path.read_text())
        row["status"] = "failed"
        put(self.root, "executions/pnn-v1-exp001-456.json", row)
        with self.assertRaisesRegex(CatalogError, "noncompleted execution"):
            build_catalog(self.root)

    def test_duplicate_json_keys_and_unknown_source_fail_closed(self):
        path = self.root / "results/exp006_demo.json"
        raw = path.read_text()
        path.write_text(raw.replace('"backend": "qant-cpu",',
                                    '"backend": "qant-cpu", "backend": "spoof",'))
        with self.assertRaisesRegex(CatalogError, "duplicate JSON key"):
            build_catalog(self.root)
        fixture(self.root)
        put(self.root, "results/unreviewed_experiment.json", {"experiment_id": "other"})
        with self.assertRaisesRegex(CatalogError, "unclassified root results JSON"):
            build_catalog(self.root)

    def test_unapproved_draft_promotion_fails_closed(self):
        path = self.root / "research_queue/v3_protocols/first_standalone_draft.json"
        draft = json.loads(path.read_text())
        draft["execution"]["training_runs"] = 1
        put(self.root, "research_queue/v3_protocols/first_standalone_draft.json", draft)
        with self.assertRaisesRegex(CatalogError, "unapproved/unexecuted draft"):
            build_catalog(self.root)

    def test_missing_source_proposal_is_explicitly_null(self):
        # An auxiliary result may lack a committed proposal: never fabricate it.
        primary = self.root / "pnn-v1/results/result_w9_confirmation.json"
        put(self.root, "pnn-v1/results/result_w9_confirmation.json", {
            "schema_version": 1, "experiment_id": "PNN-v1-W9-Confirmation",
            "proposal_id": "not-committed", "backend": "qant-cpu/software-simulation",
            "dataset": "ECG200", "configuration": {"seeds": [303], "epochs": 1},
        })
        record = next(r for r in build_catalog(self.root)["records"]
                      if r["source"]["path"] == primary.relative_to(self.root).as_posix())
        self.assertIsNone(record["proposal_source"])
        self.assertEqual(record["job_link_basis"], "unlinked_auxiliary_result")

    def test_actual_repository_is_parseable_and_series_are_distinct(self):
        # This is read-only: validates the real source inventory in the PR.
        real = build_catalog(ROOT)
        ids = {r["catalog_id"] for r in real["records"]}
        self.assertIn("toolkit:result:exp030_qant_compatibility_model_v3:v1", ids)
        self.assertIn("pnn-v1:result:PNN-v1-Exp021:v1", ids)
        self.assertIn("v3:protocol_draft:v3-ecg200-width10-11-fresh-control16-draft1:v1", ids)
        self.assertGreaterEqual(real["summary"]["canonical_completed_result_records"], 56)
        self.assertGreaterEqual(real["summary"]["exact_pnn_mirror_files"], 26)


if __name__ == "__main__":
    unittest.main()
