# Global experiment catalog (#154)

> **Read-only, derived human/audit discovery index.** The primary result/proposal JSON, approval records and exact source Git revision remain authoritative. The catalog does not approve runs, authenticate GitHub Actions artifacts, connect to Groq/Q.ANT, expose a research-tool endpoint or revise frozen Dev-4 evidence.

## Access and regeneration

- Builder: `scripts/build_experiment_catalog.py` (Python 3.11+, standard library only).
- Versioned snapshot: [`public/experiment-catalog.json`](../public/experiment-catalog.json). This reflects the source bytes recorded by its `source_manifest_sha256`; check freshness after subsequent result merges.
- Validation: `python -m unittest discover -s tests -p 'test_build_experiment_catalog.py' -v`.
- Regenerate locally from repository root: `python -m scripts.build_experiment_catalog --output public/experiment-catalog.json`.
- Check that the committed snapshot matches the currently checked-out source files, without writing: `python -m scripts.build_experiment_catalog --check`.
- Generate a throwaway snapshot: `python -m scripts.build_experiment_catalog --output /tmp/experiment-catalog.json`.

Regenerate and review the derived snapshot whenever newly approved results or their associated proposal/job/execution records are added. The PR-only offline catalog validation workflow tests the builder and checks freshness when a snapshot is present; it does **not** automatically approve experiments, alter an existing guarded execution workflow, or publish future catalogs on every data push. A user relying on a stored snapshot should run `--check` against the desired checkout.

## Scope and record identities

| Input | Handling |
|---|---|
| `results/expNNN_*.json` | Canonical original Toolkit-discovery results, irrespective of success/failure against their scientific criteria. |
| `pnn-v1/results/*.json` | Canonical separately numbered PNN-v1 results, including distinct auxiliary/confirmation experiments. |
| `results/result*.json` | Exact byte-identical mirror of the same named file in `pnn-v1/results/`; added under `mirrored_sources` and **not** counted as another completed experiment. Missing/divergent mirrors fail closed. |
| Original Toolkit and PNN-v1 committed proposal files | Separately typed `proposal` records; a still-proposed source document must not relabel an already recorded result as unexecuted. |
| `research_queue/v3_hypotheses/registry.json` | Hypothesis **metadata only**, clearly `proposed_unexecuted`. No model proposal text, prospective outcome, other-strategy feedback or future result is copied. |
| `research_queue/v3_protocols/first_standalone_draft.json` | Metadata only, marked `draft_unapproved_unexecuted`; a changed execution/approval state causes validation failure rather than silent promotion. |
| `research_queue/jobs/*.json`, `executions/*.json` | Referenced by recorded job and execution identifiers where a link exists. Filenames, job IDs, run IDs and recorded Git commit formats are checked. Some older/auxiliary links use naming conventions and are explicitly labeled **not artifact-attested**. |

Every record has a composite `catalog_id`, for example `toolkit:result:exp021_qant_accumulation_scaling_law:v1` versus `pnn-v1:result:PNN-v1-Exp021:v1`. `record_kind` distinguishes `result`, `proposal`, `hypothesis` and `protocol_draft`. `source_schema_version` is null if absent from the original file; the trailing `v1` is the **catalog record version**, not a guessed source schema.

Each source pointer includes the relative path, its SHA-256 over the checked-out bytes and the standard Git blob SHA-1 for those same bytes. These identify content but do **not** independently establish that a claimed experiment actually ran, that an approval existed at execution time, or that a GitHub Actions artifact was authentic. The recorded `execution_records` and `job_link_basis` make that distinction visible. Individual metrics remain in the original result files, not silently rounded or combined into cross-protocol rankings.

## Frozen evidence and information isolation

**The global catalog is deliberately not Research Memory.** Its code is not imported by `scripts/research_memory_v3.py`, and it is not registered as a tool for the v3 model. It does **not** scan frozen `research_queue/benchmarks/` feedback, Dev-4 strategy contexts, candidate holdout data, approval history, or transient `local_results/`. A human may use it to locate already published results across the lab; a Dev-4 replay/selector must still receive only its exact frozen initial snapshot, its own proposal/status ledger and its own previously completed permitted outcomes. In particular, discovering an older width-10 ECG200 study must not make it a retrospective input to the frozen own-strategy GPT-OSS memory.

The catalog reports **committed source evidence and metadata**, not actual photonic-chip latency, energy, throughput or optical performance. The Q.ANT CPU/software-simulation backend is not the physical photonic processor.

## Failure policy

The builder raises an error instead of skipping ambiguous research data: duplicate source JSON keys, unexpected result names, duplicate composite identities, source/file ID mismatches, root-only PNN copies, changed mirror bytes, mismatched job/proposal associations, malformed execution/run IDs, or v3 draft/hypothesis records masquerading as executed work. Tests include intentionally doctored fixtures and a read-only scan of the actual repository.

**Intentional exclusions:** root `automation_smoke_test.json` and `result_schema.example.json` are infrastructure/example data; other unclassified JSON under root `results/` is rejected pending review. The separate Dev-4 source-pinned selection benchmark remains governed by [`dev4_selection_protocol.json`](../research_queue/benchmarks/dev4_selection_protocol.json) and its own validators. See the [current systems and protocol map](current-systems-and-protocols.md).
