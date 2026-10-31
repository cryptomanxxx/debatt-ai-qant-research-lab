"""No-compute CL-3F scientific governance contracts."""
import contextlib
import io
import json
import unittest
from unittest.mock import patch

from integrations.continual_learning.cl3f_scientific_gate_audit import (
    EVIDENCE, main, review_matrix,
)


class CL3FGateAuditContract(unittest.TestCase):
    def test_all_eight_blockers_remain_open(self):
        audit = review_matrix()
        self.assertEqual("NO_GO", audit["decision"])
        self.assertFalse(audit["training_allowed"])
        self.assertFalse(audit["compute_approved"])
        self.assertEqual(8, audit["blockers_open"])
        self.assertEqual(set(EVIDENCE), {x["blocker_id"] for x in audit["matrix"]})
        for row in audit["matrix"]:
            self.assertFalse(row["scientifically_validated"])
            self.assertEqual("OPEN_REQUIRES_INDEPENDENT_REVIEW", row["status"])
            self.assertTrue(row["candidate_evidence_files_present"])

    def test_cli_never_offers_training(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(0, main(["--review-matrix"]))
        self.assertFalse(json.loads(out.getvalue())["training_allowed"])
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as ctx:
                main(["--train"])
        self.assertEqual(2, ctx.exception.code)

    def test_cannot_accept_false_approval_or_closed_blocker(self):
        from integrations.continual_learning import cl3f_scientific_gate_audit as module

        valid = json.loads(module.READINESS.read_text(encoding="utf-8"))
        approved = dict(valid, compute_approved=True)
        with patch.object(module.READINESS.__class__, "read_text", return_value=json.dumps(approved)):
            with self.assertRaises(ValueError):
                review_matrix()
        closed = json.loads(json.dumps(valid))
        closed["mandatory_blockers"][0]["closed"] = True
        with patch.object(module.READINESS.__class__, "read_text", return_value=json.dumps(closed)):
            with self.assertRaises(ValueError):
                review_matrix()


if __name__ == "__main__":
    unittest.main()
