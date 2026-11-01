# CL-3S: GitHub Actions inline charts + optional Pages report

The primary `CL Research Visualization (static, no training)` workflow
now writes **Unicode bar charts and sparklines directly into the GitHub
Actions job Summary** through `GITHUB_STEP_SUMMARY`. They are rendered as
ordinary Markdown, so you can see them on your Android tablet without a
ZIP download, external image URL or GitHub Pages setup.

The summary includes two distinct phases (hypothetical training update
and hypothetical inference) and an activity sensitivity curve across
0–16 active bits. Bars and curves show **assumed logical access payload
bytes, not joules**. Existing SVG, HTML and JSON artifacts remain available.

## Optional full visual report via GitHub Pages

A *separate* `CL Research Charts — optional GitHub Pages publication`
workflow is deliberately manual (`workflow_dispatch` only). It creates a
standalone HTML report with full SVG plots and attempts a Pages deployment.

Before manually running it:

1. In the repository's **Settings → Pages**, enable GitHub Pages and set
   its publishing source to **GitHub Actions**.
2. Check that your repository visibility and Pages settings are compatible.
   **Pages deployment may publish the research report publicly.**
3. Then choose the Pages workflow under **Actions → Run workflow**.
4. When it succeeds, open the public `page_url` in the run summary or
   environment deployment. The page will show architecture and cost plots
   directly, with no ZIP unpacking.

The regular CL-3R workflow never publishes publicly; only the explicit
Pages workflow can deploy. The static HTML contains no external scripts,
APIs, credentials, dataset records or private training results.

Both workflows generate source-level cost *estimates only* and do not
instantiate models, train, run AI inference or measure real energy.
**CL-3A remains NO_GO for experimental training.**
