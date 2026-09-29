# AI Researcher v2 — experiment-selection measurement (phase 1)

The approved-proposal -> compiler -> human scientific approval -> separate
compute authorization -> guarded execution path already exists on dev. Do not
replace or bypass it. This change introduces **measurement before optimization**.

Run locally from repository root:

```bash
python -m unittest discover -s tests -p 'test_measure_researcher_selection.py'
python scripts/measure_researcher_selection.py --output /tmp/researcher-selection-baseline.json
```

The script reads completed structured JSON results under `pnn-v1/results`.
For each candidate with a concurrent control, it records aggregate correct
counts, paired correct-count margin, parameter saving and whether the fixed
relative gate (candidate >= control - 10) passed. Invalid/incomplete files are
listed as skipped rather than silently treated as success.

The exploratory Pareto view only compares candidates evaluated with the **same
seed list**. It is not evidence that a model generalizes across seeds, datasets
or hardware. This report is read-only: it neither generates proposals nor
approves/dispatches compute and does not call Groq.

Next phases, only after this baseline is reviewed:
1. Bind proposal SHA and approved job/result identifiers for prospective
   attribution of outcomes to researcher suggestions.
2. Pre-register equal-budget comparison against random and Bayesian selection
   on an approved bounded experiment family, with fresh paired seeds.
3. Introduce a proposal-only selector; retain the existing compiler, scientific
   review and separate human compute authorization unchanged.
