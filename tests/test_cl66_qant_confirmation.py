"""CL-66: evaluation paths agree with CL-62/CL-65 simulator code; packing round-trips."""
import base64
import hashlib
import unittest
import zlib

import numpy as np

import experiments.cl66_qant_confirmation.run as R


class CL66(unittest.TestCase):
    def test_mono_path_equals_cl62_seq(self):
        model = R.cl58.QFourier(1, 4)
        X = np.random.default_rng(0).random((5, R.cl58.D)).astype(np.float32)
        np.testing.assert_array_equal(R.mono_logits(R.SimBackend, model, X), R.cl62.variant_logits(model, X, "seq"))

    def test_hier_paths_equal_cl65(self):
        model = R.cl64.Hier(2, 28, 2)
        X = np.random.default_rng(1).random((4, R.cl58.D)).astype(np.float32)
        H = R.cl65.stage1(model, X, "seq")
        for h in (1, 8):
            np.testing.assert_array_equal(R.hier_logits(R.SimBackend, model, X, h), R.cl65.grouped_head(model, H, h, "seq"))

    def test_flip_stats(self):
        ref = np.array([0, 1, 2, 3, 4], np.uint8)
        sim = np.array([0, 9, 2, 9, 4], np.uint8)
        q = np.array([0, 9, 9, 3, 4], np.uint8)
        s = R.flip_stats(ref, sim, q)
        self.assertEqual((s["dis_ref_qant"], s["dis_ref_sim"], s["flipped_both"]), (2, 2, 1))
        self.assertAlmostEqual(s["jaccard"], 1 / 3)

    def test_pack_round_trip(self):
        preds = {"b": np.arange(7, dtype=np.uint8), "a": np.ones(7, np.uint8)}
        keys, data, digest = R.pack(preds)
        raw = zlib.decompress(base64.b64decode(data))
        self.assertEqual(hashlib.sha256(raw).hexdigest(), digest)
        np.testing.assert_array_equal(np.frombuffer(raw, np.uint8)[:7], preds[keys[0]])


if __name__ == "__main__":
    unittest.main()
