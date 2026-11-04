"""CL-73: encoder gradients, Q.ANT path arithmetic and retrieval metrics."""
import unittest

import numpy as np

import experiments.cl73_search_embeddings.run as R


def loss_fn(p, ids, n):
    e = R.forward(p, ids)
    return R.contrastive(e[:n], e[n:])[0]


class CL73(unittest.TestCase):
    def test_gradients_match_finite_differences(self):
        p = {k: v.astype(np.float64) * (5 if v.ndim == 2 else 1) for k, v in R.init_params(0).items()}
        rng = np.random.default_rng(1)
        seqs = [[R.CLS] + rng.integers(3, 60, size=int(rng.integers(3, 9))).tolist() for _ in range(6)]
        ids = R.pad(seqs)
        e, c = R.forward(p, ids, cache=True)
        _, dq, dd = R.contrastive(e[:3], e[3:])
        g = R.backward(p, np.concatenate([dq, dd]), c)
        for name in ("Wq0", "Wk1", "Wv0", "Wo1", "W10", "W21", "g10", "be21", "gf", "Wp", "E", "P"):
            flat = p[name].reshape(-1)
            idx = rng.choice(np.flatnonzero(np.abs(g[name].reshape(-1)) > 1e-8), 3)
            for j in idx:
                old = flat[j]
                flat[j] = old + 1e-6
                up = loss_fn(p, ids, 3)
                flat[j] = old - 1e-6
                dn = loss_fn(p, ids, 3)
                flat[j] = old
                self.assertAlmostEqual((up - dn) / 2e-6, g[name].reshape(-1)[j], delta=1e-5 + 1e-3 * abs(g[name].reshape(-1)[j]), msg=name)

    def test_qant_path_is_bf16_with_float32_accumulation(self):
        a = np.array([[1.0 + 2 ** -10, 1.0]], np.float32)
        b = np.array([[1.0], [2 ** -12]], np.float32)
        path = R.Path_("qant")
        self.assertEqual(float(path.mm(a, b)[0, 0]), 1.0)  # operand rounded, tiny term lost in the bf16 output
        self.assertEqual(float(R.Path_("f32").mm(a, b)[0, 0]), float((a @ b)[0, 0]))

    def test_padding_does_not_change_embeddings(self):
        p = R.init_params(3)
        s = [R.CLS, 5, 6, 7]
        e1 = R.forward(p, R.pad([s]))
        e2 = R.forward(p, R.pad([s, s + [8] * 10]))[:1]
        np.testing.assert_allclose(e1, e2, atol=1e-6)

    def test_rank_metrics(self):
        S = np.array([[0.9, 0.1, 0.2], [0.5, 0.4, 0.6]])
        m = R.rank_metrics(S, np.array([0, 1]))
        self.assertEqual(m["recall_at_1"], 0.5)
        self.assertAlmostEqual(m["mrr"], (1 + 1 / 3) / 2)

    def test_task_split_is_disjoint_and_fixed(self):
        paras = R.load_corpus()
        tr, te, q, docs = R.build_task(paras)
        self.assertFalse(set(tr) & set(te))
        self.assertEqual(len(q), 4 * len(te))
        for i, qw, dw in q[:20]:
            self.assertEqual(len(qw) + len(dw), len(paras[i]))
        self.assertEqual(R.build_task(paras)[2][:5], q[:5])

    def test_parameter_count_is_tier0(self):
        self.assertLess(R.n_params(R.init_params(0)), 1_000_000)


if __name__ == "__main__":
    unittest.main()
