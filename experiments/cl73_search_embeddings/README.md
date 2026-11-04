# CL-73 — Search embeddings from a small BERT-style encoder on the simulated Q.ANT path

**Tier 0** under [docs/RISK_TIERS.md](../../docs/RISK_TIERS.md): under 1M parameters, text already in the repository, no network, about 9 minutes locally. Follows up [CL-72](../cl72_application_mapping/README.md), which ranked short-sequence transformer encoders first. [Predictions](PREDICTIONS.md) were pushed before any training (commit `2b20b93`). Physical energy: `NOT_MEASURED`.

Reproduce from the repository root: `python -m experiments.cl73_search_embeddings.run`, which writes [`results.json`](results.json). The corpus is frozen in `corpus.json.gz` (`build_corpus.py`).

## Question

Semantic search embeds queries and documents with a transformer encoder and ranks documents by cosine similarity. **Does running every matrix product of the encoder on the Q.ANT `linear_fprop` path change which documents are retrieved?**

## Method

Pretrained BERT-base (110M parameters, download) would be Tier 1. Instead, a BERT-style encoder is trained from scratch.

- **Corpus:** 1,266 Markdown paragraphs from this repository at commit `27970ec`.
- **Task:** inverse cloze retrieval. The query is a 12–20-word span, and the target is its paragraph with that span removed.
- **Evaluation:** 253 held-out paragraphs with 4 queries each (1,012 queries), ranked against all 1,266 documents.
- **Encoder:**
  - 2 pre-LN layers, d = 64, 4 heads, feed-forward 256 with GELU, mean pooling and a projection;
  - 368,640 parameters;
  - trained with an in-batch contrastive loss, 40 epochs, seeds 11/22/33.
- **Q.ANT path:**
  - every matrix product, including the dynamic attention products QKᵀ and AV, uses the CL-70 model of `linear_fprop`: bf16 operands and output, float32 accumulation;
  - the bias is added in bf16;
  - LayerNorm, GELU, the residual adds and pooling run on the host in float32.

The full design is in [PREDICTIONS.md](PREDICTIONS.md).

## Results

Means over 3 seeds and 1,012 queries:

| Variant | Recall@1 | Recall@10 | MRR | Top-1 agreement with float32 | Top-10 overlap |
|---|---|---|---|---|---|
| Chance | 0.08% | 0.8% | — | — | — |
| Encoder, untrained | 0.96% | 4.8% | 0.026 | — | — |
| BM25 | **5.53%** | **24.5%** | **0.121** | — | — |
| Encoder, float32 | 2.50% | 18.0% | 0.078 | 100% | 100% |
| Encoder, Q.ANT path | 2.54% | 18.1% | 0.079 | 99.4% | 99.6% |
| Encoder, mixed (float32 index, Q.ANT queries) | 2.50% | 18.1% | 0.078 | 99.5% | 99.6% |
| Encoder, Q.ANT + bf16 softmax | 2.54% | 18.1% | 0.079 | 99.4% | 99.6% |

- **Embedding fidelity.** The mean cosine between float32 and Q.ANT-path embeddings of the same text is 0.999996, with a minimum of 0.999986. That corresponds to about 0.3% relative error, the bf16 rounding level.
- **Per seed.** The largest change in Recall@1 is +0.10 pp (seed 33). The largest change in MRR is +0.0006.
- **Margins.** About 6 in 1,000 queries change their top-1 document. The float32 median margin between the best and second-best document is 0.014.
- **bf16 softmax** gives results identical to host softmax, bit for bit. This could have been derived: on the `linear_fprop` path, the attention probabilities are rounded to bf16 anyway when they enter the AV product.
- **Dynamic MACs.** At this corpus's mean length of 64 tokens, 17% of the small encoder's MACs are dynamic attention products. BERT-base at 128 tokens has 2.7% (CL-72), because a small width d shrinks the weight products but not the attention products.

## Predictions

| # | Prediction | Outcome |
|---|---|---|
| 1 | Q.ANT and mixed paths: \|ΔRecall@1\| ≤ 1 pp and \|ΔMRR\| ≤ 0.01 per seed, cosine ≥ 0.999, top-1 agreement ≥ 95% | ✓ ≤ 0.10 pp, ≤ 0.0006, 0.999996, ≥ 99.3% |
| 2 | bf16 softmax within 0.5 pp of host softmax | ✓ identical (derivable, see above) |
| 3 | Trained Recall@1 ≥ 5× untrained and far above chance | ✗ 2.6× on average (4.5×, 4.7×, 1.6× per seed); 32× chance |
| 4 | BM25 beats the trained encoder on Recall@1 | ✓ 5.5% against 2.5% |

**3 of 4 held.** Training loss fell from 4.3 to 0.4–0.9 while held-out recall stayed low. The encoder overfits roughly 1,000 training paragraphs, so it remains a weak retriever. The hyperparameters were fixed in advance and not tuned afterwards.

## What this does and does not show

- **It shows that the search ranking survives the Q.ANT `linear_fprop` arithmetic.** A transformer encoder's embeddings, and therefore its rankings, are essentially unchanged on the simulated Q.ANT path. That includes the case where the index is built in float32 and only queries run on Q.ANT. It is a necessary condition for using Q.ANT for embedding search, and it follows mostly from bf16 rounding with float32 accumulation, as derived in the predictions.
- **It does not show a useful search model.** The encoder is far weaker than BM25 here. A real test of retrieval quality needs a pretrained encoder (Tier 1 download, more than 1M parameters).
- **Simulator only.** No Q.ANT CPU-backend call was made. A confirmation run on the CPU backend would be Tier 2. Speed, latency and energy on photonic hardware are unknown.
- **Unmeasured near-ties.** Larger, better-trained encoders have different score margins. The share of top-1 changes (0.6%) may differ for them.
