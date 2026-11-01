# CL-3R: GitHub Actions static research visualizations

**NO_GO / NO TRAINING / NO MEASURED ENERGY.**

This report is generated using only the existing pure source-level CL-3E
(training-step) and CL-3N (standalone inference) cost estimators. It does
not initialize or update a model, generate synthetic datasets, run
predictions, contact external compute or make physical energy claims.

## Open on Android or desktop

1. Open the repository's **Actions** tab.
2. Choose **CL Research Visualization (static, no training)**.
3. Choose **Run workflow** on the desired reviewed branch.
4. When finished, open the run's **Summary** tab for a phase-separated
   comparison table.
5. Under **Artifacts**, download `cl3r-static-research-visuals`, unzip it
   and open `report.html` in a browser. SVG images and the raw
   `projections.json` are included for reuse.

The HTML file is standalone and includes the architecture diagram,
phase-separated bar comparison at two active inputs, and two sensitivity
charts for 0–16 active inputs. No external CDN, script or login is needed
to view the downloaded report. Report generation uses only the Python
standard library; no Matplotlib dependency is required.

## Scope and cautions

- **Training** is hypothetical `project_training_only(...step_count=1)`
  under an assumed fixed activity, not a trained system.
- **Inference** is a distinct source-loop `estimate_prediction()`
  under the same hypothetical activity, not a prediction call.
- Logical bytes use explicitly assumed 8-byte parameters, 1-byte inputs
  and 2-byte active indices. They are not CPU time, measured RAM traffic,
  physical energy or a photonic processor result.
- This overview intentionally does not use CL-3O/P/Q's synthetic-task
  distributions: it displays controlled *input activity sensitivity* to
  make the architecture relationships easier to understand.
- Future real accuracy/glömska charts will only be added after a
  separately reviewed, explicitly authorized experiment.

This manual workflow is **not triggered automatically** by a PR merge.
The CL-3A NO_GO decision remains unchanged.
