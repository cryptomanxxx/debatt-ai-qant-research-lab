# CL-73 predictions — search embeddings from a small BERT-style encoder on the simulated Q.ANT path (written before any training or evaluation)

Committed and pushed before any training or evaluation. Informal predictions, not a formal preregistration. A follow-up to CL-72, which ranked short-sequence transformer encoders as the best candidate application.

## Question

Semantic search encodes queries and documents with a transformer encoder and ranks documents by cosine similarity. **Does running every matrix product of such an encoder on the Q.ANT `linear_fprop` path change which documents are retrieved?**

## Why not real BERT

Pretrained BERT-base has 110M parameters and must be downloaded. Under [RISK_TIERS](../../docs/RISK_TIERS.md), that is Tier 1. This run stays at Tier 0: a BERT-style encoder of under 1M parameters, trained from scratch on text already in the repository.

## Design

- **Corpus.** 1,266 Markdown paragraphs of 30–400 words from the repository at commit `27970ec`, near-duplicates removed and each truncated to 120 words (`build_corpus.py`, frozen in `corpus.json.gz`).
- **Task.** Inverse cloze (Lee et al., 2019). The query is a random span of 12–20 consecutive words. The document is its paragraph with that span removed. A fixed 20% of paragraphs (split seed 0) are held out. Each gets 4 queries. The index is all 1,266 documents, with every held-out document's query span removed. Only held-out queries are scored, so test paragraphs are never seen in training.
- **Encoder.**
  - Word vocabulary: the 4,000 most frequent training words, plus UNK and CLS.
  - Two pre-LN transformer layers: d = 64, 4 heads, feed-forward 256 with GELU, learned positions, max length 128.
  - Masked mean pooling, a 64×64 projection and L2 normalisation.
  - About 0.37M parameters. The same encoder embeds queries and documents.
- **Training.** In-batch softmax contrastive loss at temperature 0.05. Batch 64, Adam lr 1e-3, 40 epochs. A fresh span is drawn every epoch. NumPy with hand-written backward pass, checked by finite differences in the tests. Seeds 11/22/33.
- **Q.ANT path.**
  - Every matrix product goes through the CL-70 model of `linear_fprop`: bf16 operands and output, float32 accumulation. That includes the dynamic attention products QKᵀ and AV.
  - Bias is added in bf16 (`add_bias_fprop`, exact).
  - LayerNorm, GELU, the residual adds, pooling and normalisation run on the host in float32.
  - Attention softmax in two variants: host float32, or a bf16-rounded output as a model of `softmax_fprop` (CL-70: one bf16 step off in 38% of values).
- **Variants scored.**
  - float32;
  - Q.ANT, with queries and index both on the Q.ANT path;
  - mixed, with the index built in float32 and queries on Q.ANT (an index built once on a GPU, queried on photonics);
  - Q.ANT + bf16 softmax.
- **Baselines.** Chance; BM25 (k1 = 1.2, b = 0.75); the same encoder untrained.
- **Metrics.**
  - Recall@1, Recall@10 and MRR over about 1,000 held-out queries;
  - top-1 agreement with float32;
  - mean top-10 overlap;
  - mean cosine between the float32 and Q.ANT embeddings of the same text.

## What can be derived on paper

bf16 has a relative rounding error of at most 2⁻⁹ ≈ 0.2%. Float32 accumulation adds almost none (CL-70: 0.32% flat). Through two layers, the embedding error should stay at the 0.1–1% level. That gives a cosine to float32 of about 1 − 10⁻⁴ or better. Rankings can then change only where two documents score within about 0.01 of each other. So the fidelity prediction below is close to derived, and the run mainly measures how often such near-ties occur.

The share of static-weight MACs in this tiny encoder can also be derived: for d = 64, feed-forward 256 and sequence L, the dynamic share is 2L / (768 + 2L), which is about 25% at L = 128. That is far worse than BERT-base's 2.7% (CL-72), because width d sets the weight products but not the attention products. It is reported, not predicted.

## Predictions

1. **Fidelity.** On the Q.ANT path and the mixed path, |ΔRecall@1| ≤ 1 pp and |ΔMRR| ≤ 0.01 against float32 for every seed. Mean embedding cosine to float32 ≥ 0.999. Top-1 agreement ≥ 95%.
2. **bf16 softmax adds little:** Q.ANT + bf16 softmax is within 0.5 pp Recall@1 of Q.ANT with host softmax.
3. **The encoder learns:** trained Recall@1 is at least 5× the untrained encoder's, and far above chance (1/1,266).
4. **BM25 wins:** on a corpus this small, with no pretraining, BM25 has higher Recall@1 than the trained encoder.
