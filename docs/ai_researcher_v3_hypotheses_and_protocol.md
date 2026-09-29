# AI Researcher v3 — original-live hypothesis register and offline study draft

## Why keep proposals separate?

AI Researcher v3 now has **two real generated hypotheses** from successful manually
approved Groq sessions. Neither is a measured result, a selected experiment,
nor evidence of any accuracy increase. Register their source and exact model
proposal without placing them in `research_memory_v3.py`: that module remains
read-only, strategy-isolated and limited to independently pinned completed
Dev-4 Round 1/2 evidence.

| Hypothesis ID | Run / GitHub Actions artifact | Proposed width | Critical review | Status |
|---|---|---:|---|---|
| `v3-live-36597959269-width11` | [36597959269](https://github.com/cryptomanxxx/debatt-ai-qant-research-lab/actions/runs/36597959269), artifact 11047390153, member `research_session.json` | 11 | None in original workflow | `proposed_untested` |
| `v3-reviewed-36601207596-width10` | [36601207596](https://github.com/cryptomanxxx/debatt-ai-qant-research-lab/actions/runs/36601207596), artifact 11048933655, member `reviewed_research_session.json` | 10 | Same GPT-OSS model, fresh-context `testable_with_caveats` | `proposed_untested` |

Source report **raw JSON SHA-256**: width 11 =
`8d436214e35b45fb713b912b64eeee4bd1d10f36667ec0faa7019e5771f9f6a1`;
width 10 =
`b4c40fa679c573e61eef5edba38301071ef638d3c6ad5380b35749b5d4064fe9`.

The registry `research_queue/v3_hypotheses/registry.json` stores the *actual seven-field
proposal text* copied from those source artifacts, observed evidence IDs,
proposer-step SHA-256, proposal's canonical SHA-256, raw report SHA-256, exact
GitHub run and artifact IDs, commit SHA and review metadata. The expected
fingerprints are checked separately by `scripts/v3_hypothesis_registry.py`.
Changing a source, introducing fabricated measurements, renaming it as
completed or changing cited experiment IDs fails closed. Future genuine
hypotheses require intentionally adding new pins and review.

Source artifacts have GitHub Actions retention windows; to independently
recheck a **locally downloaded, original, unmodified** report before expiry,
use the optional `verify_original_report_bytes(raw_json_bytes, entry)` API.
It compares the raw source's SHA-256, the audited model-proposal step, proposal
text, evidence IDs, review linkage where present and zero training runs. It
does not download artifacts, connect to Groq or import training code. The
registry retains the exact proposal text and source fingerprints for offline
planning even when Actions artifacts expire.

## Reproduce the read-only plan

From the repository root, on Python 3.11:

```bash
python -m unittest discover -s tests -p 'test_v3_hypothesis_registry.py' -v
python -m scripts.v3_hypothesis_registry > /tmp/v3_offline_study_draft.json
```

This creates a local, machine-readable **unapproved and unexecuted** draft
without GitHub dispatch, model API requests or a training import. Both
registered proposals are independently replayed through
`build_evaluation_plan` against the exact pinned completed Research Memory
rows. A proposed width does **not** appear in completed Research Memory.

For optional independent provenance check of downloaded GitHub Actions ZIPs:

```python
import zipfile
from scripts.v3_hypothesis_registry import (
    load_hypotheses, verify_original_report_bytes,
)

pairs = [
    ("dev-v3-first-live-36597959269.zip", "research_session.json"),
    ("dev-v3-reviewed-live-36601207596.zip", "reviewed_research_session.json"),
]
for entry, (archive, name) in zip(load_hypotheses()["entries"], pairs):
    with zipfile.ZipFile(archive) as z:
        assert verify_original_report_bytes(z.read(name), entry)
```

The two ZIPs must first be obtained from their linked original Actions runs;
the script makes no download/network requests.

## Distinguish history, new hypotheses and a future study

The *completed* own-strategy historical data consists only of:
width 9 → 445/500 correct, 4,812 parameters (round 1), and
width 8 → 438/500 correct, 4,278 parameters (round 2).
The software-simulation historical control was width 16 →
452/500 correct and 8,550 parameters. This historical source is
`qant-cpu/software-simulation` — **not a photonic hardware measurement**.

The protocol *draft* reproduces the old comparison settings from both frozen
rounds: ECG200, 100 epochs, seeds 301–305, learning rate 0.001,
batch size 32, frequencies [1, 2]. It proposes, but does not authorize,
testing both untested widths 10 and 11 against a **fresh, seed-paired width-16
control** in a standalone v3 study. Do not assume a freshly trained control
will reproduce the historic 452/500.

Potential future measurements include correctness and parameter counts
**per seed** for each candidate and the freshly paired control, as well as
paired differences; retain raw results for assessing seed-level variability.
Before any actual execution, the team must separately preregister the
primary endpoint, uncertainty method, objective/thresholds, backend, job
count and compute authorization.

The machine plan distinguishes the *historic* gate of 442/500, exact
*historic* control tie at 452/500, strict historic outperformance at 453/500,
and strict improvement over the best observed own-strategy candidate
at 446/500. These are descriptive comparator thresholds **not prospective v3
success rules**. “Approaches the control” has no numeric definition unless
it is separately preregistered. Parameter count and accuracy should be
measured rather than extrapolated from the two historical points.

## Hard safety boundary

There is no new experiment file in the training queue, no GitHub workflow
dispatch, no scheduled or paid Groq call, and **zero** Q.ANT CPU or optical
training runs. Dev-4 frozen records and workflows are unchanged. There is
no photonic claim or automatic promotion of a proposed hypothesis into
completed evidence. PR review and merge do not grant execution approval.
The existing manual reviewed-live workflow only runs after its own separately
provided explicit approval; this PR merely adds an offline provenance
preflight test.
