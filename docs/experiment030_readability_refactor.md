# Experiment 030 — audited readability-only refactor (PR #152)

## Scope

Claude Code identified `experiments/030_qant_compatibility_model_v3/run.py`
as difficult to review: the original 54-line source placed many sequential
operations on single lines and compressed class definitions, metrics and
serialization. This change expands that source into conventional indentation,
multiline expressions/dictionaries and explanatory comments.

**This PR intentionally does not redesign or rename the experiment.** It
retains the historical `FB`, `H`, `npar`, `SEEDS`, `ARCH`, `K`, local
identifiers, checkpoint/data paths, job environment input, all call ordering,
architectures, RNG seeds, Fourier frequencies, train/test subset construction,
bfloat16 conversion boundaries, optimizer/training loop, progress reporting,
metrics, and output JSON schema. Renaming identifiers or extracting helpers
would create a separate semantic-review surface; this PR is only a safe first
layout pass. These names can be revisited in a separately reviewed change.

## Equivalence preflight

The initial `main` source used for the refactor was independently checked
against its Git blob:
`64b1465edbaf65b57a33ab67a083cab522bba9d4`.

AST serializations are Python-version sensitive. For the **same original
source**, `ast.dump(ast.parse(source), annotate_fields=True,
include_attributes=False)` has these interpreter-specific SHA-256 pins:

| Interpreter | Original-source AST SHA-256 |
|---|---|
| Python 3.11 (CI) | `46102e911ec43c0ed95c9950abb4f8ce3b4ed39975dcaa5e73182dcfa9bb6852` |
| Python 3.13 (local preflight) | `870783abe27a066ea932deb0a7515aa31b4c412fad4c844ad0ec26383b04a05a` |

When the checkout retains the original Git blob (as PR CI does using a
two-commit shallow checkout), the offline test also compares the original
and reformatted ASTs **directly under the same interpreter**, rather than
relying on hashes computed across interpreter versions.

The reformatted source has **exactly the same AST**, not merely the same
strings appearing somewhere in a file. This comparison includes all executable
statements, expression order, literal constants, classes, function bodies,
training/evaluation code, output keys, and imports. Comments, indentation and
line numbers are intentionally excluded. The dedicated regression test
`tests/test_experiment030_refactor_contract.py` also enforces reasonable
line lengths and absence of semicolon chains.

This is a source-equivalence safeguard, **not** a claim of new numerical
reproduction on the CPU backend. No previously generated experimental result
or stored result JSON was changed.

## How to verify without initiating compute

```bash
python -m unittest discover -s tests -p 'test_experiment030_refactor_contract.py' -v
```

The CI is extended to trigger when Experiment 030 itself changes, and executes
this test as part of the standard `Validate Research Safety Fixes` job.
It only reads and parses the file, compiling the syntax tree without
executing it. **Do not import or invoke the experiment's run.py for this
test:** the legacy top-level runner is intentionally live-executing and loads
FashionMNIST / trains models if provided a job config.

This PR makes no Groq request, does not dispatch a workflow, does not run
Q.ANT CPU/software-simulation or photonic hardware, changes no frozen Dev-4
record, adds no permission, and does not authorize a subsequent experiment.
