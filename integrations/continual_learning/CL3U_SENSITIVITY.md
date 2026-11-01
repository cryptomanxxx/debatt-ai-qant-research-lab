# CL-3U — Static sensitivity analysis (NO TRAINING)

## Question

Are the CL-3T sparse-vs-dense assumed logical-byte crossovers robust when input/hidden dimensions and hypothetical packed payload widths vary?

This is a **counterfactual analytic sensitivity study**, not an experiment with resized trained neural networks.

## Bounded scenario grid

- Network dimensions (input × hidden × output): **8×4×2**, **16×8×2**, **32×16×2**.
- Assumed packed byte widths (parameter / input / active index): **8/1/2**, **4/1/2**, **2/1/4**.
- Activity: every integer from zero through the scenario's input size.
- Comparisons: dense SGD versus local hybrid, sparse local and shared-index sparse; **training and inference are evaluated separately**.
- Strict inequality defines an advantage. Equality is **not** an advantage.

The 16×8×2 calculations are asserted equivalent to the independently implemented CL-2 and CL-3E/N source-loop estimators at every activity level for multiple width assumptions. **Other shapes are algebraic extrapolations of these Python loop patterns.** They have not been implemented, trained, timed or validated as correct ML architectures. This is a robustness probe of an *accounting model*, not a claim of scalable hardware savings.

## Data and limits

The manual **CL Research Visualization (static, no training)** GitHub Actions workflow renders the CL-3U comparison table inline and includes `sensitivity.json` in its downloadable artifact. The `Continual Learning CPU Contracts` workflow tests the pure arithmetic and the 16×8×2 cross-check. Neither workflow trains models.

The proxy includes logical packed parameter accesses, modeled input visits and active-index accesses. It does **not** measure DRAM traffic, caches, Python allocation, activation memory, nonlinearities, sensor/optical conversion, runtime, accuracy, forgetting or electrical energy. Broader shapes could change loop structure and optimization choices; no power/cost estimate should be inferred from these results.

This analysis is relevant only to **H1 accounting assumptions**, not H2 continual learning performance or H3 energy–quality tradeoffs. The scientific CL-3A **NO_GO** gate remains unchanged; any training requires explicit, separate authorization.
