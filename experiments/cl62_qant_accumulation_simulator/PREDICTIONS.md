# CL-62 predictions — does low-precision accumulation explain the Q.ANT mismatch? (written before any run)

Committed and pushed before any run. Informal predictions, not a formal preregistration.

## Question

In [CL-58](../cl58_qant_quantization_aware_training/README.md), the Q.ANT CPU backend disagreed with the float32 reference on 260–535 of 10k Fashion-MNIST test images (18 runs). A simulator that rounds operands and cosine outputs to bfloat16, but sums in float32, explained only 2–8 of them. **Does adding low-precision accumulation and output rounding to the simulator reproduce the Q.ANT disagreement counts?**

## Method

1. **Retrain the 18 CL-58 models** (grids 4/8 × 3 arms × seeds 11/22/33) with the unchanged CL-58 code. Training is deterministic: CL-58's reference and simulator counts were identical on two machines.
2. **Evaluate each model with these simulator variants,** fixed in advance. All of them start from CL-58's operand rounding: x, φ, A, b and cos rounded to bfloat16. Terms are ordered pixel-major (i, then k) within each logit.
   - `out`: CL-58 simulator, then the KAN output rounded to bfloat16, plus bfloat16 bias, rounded to bfloat16.
   - `seq`: each term bf16(cos · A), summed sequentially with the running sum rounded to bfloat16 after every addition, then the `out` rounding.
   - `pair`: the same terms summed in a pairwise tree, with every partial sum rounded to bfloat16, then the `out` rounding.
   - `block32`, `block256`: float32 sums within contiguous blocks of 32 or 256 terms, each rounded to bfloat16, then summed in float32, then the `out` rounding.
3. **Compare against Q.ANT.** For each run, count disagreements between the reference and the variant, and compare them with the archived reference-vs-Q.ANT count. Report mean counts per grid and the Pearson correlation across the 18 runs.

**Limitation:** CL-58 archived only counts, not per-image Q.ANT predictions. Matching counts are therefore necessary, not sufficient. They cannot show that the variant flips the *same* images. That needs a new Q.ANT run that saves predictions (Tier 2).

## Derived before running

- **bfloat16 resolution.** bfloat16 has an 8-bit significand, so its unit roundoff is u = 2⁻⁹ ≈ 0.002.
- **Sequential accumulation.** Its error bound grows like n·u, about 6 for grid 4 (n = 3,136) and 12 for grid 8 (n = 6,272). Once the running sum is large, small terms are lost.
- **Pairwise accumulation.** Its error bound grows like log₂(n)·u, about 0.024–0.026 relative.
- **Consequence.** Sequential bfloat16 accumulation should be far noisier than pairwise.

## Predictions

1. The retrained models reproduce CL-58's reference-vs-simulator counts exactly in all 18 runs.
2. Output rounding alone (`out`) gives less than 30% of the Q.ANT disagreement counts.
3. `seq` gives more than 3× the Q.ANT counts. In other words, Q.ANT does not accumulate naively in bfloat16.
4. **Main bet:** at least one of `pair`, `block32` or `block256` has mean disagreement counts within ±20% of Q.ANT for **both** grids.
5. For that variant, the correlation with the Q.ANT counts across the 18 runs is at least 0.8.
