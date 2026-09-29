# Debatt-AI Q.ANT Research Lab

An open, experimental research project exploring **neural-network architectures for Q.ANT's Native Computing Toolkit** and, ultimately, photonic computing.

The central question is not simply:

> How can an existing neural network be executed faster on new hardware?

It is:

> **What should neural networks look like when they are designed around the computational primitives of a different kind of processor?**

This repository contains the experiments, results and research automation behind the public Debatt-AI Q.ANT Research Lab.

## Current status

**Start here:** [Current systems and protocols](docs/current-systems-and-protocols.md) is the maintained map of active vs historical systems, frozen evidence, protocol owners and human approval. This README is an overview, not an execution contract.

The project is an early-stage research programme. Experiments currently run on **GitHub Actions standard CPU runners** using the **Q.ANT Native Computing Toolkit CPU backend**.

The Q.ANT CPU backend is software that allows Native Computing Toolkit operations to be developed and evaluated without access to the photonic processor. **Q.ANT is not providing the CPU compute used by these experiments; GitHub Actions currently provides the CPU, RAM and runtime.**

Results from the CPU backend can therefore tell us about software execution, numerical behaviour, compatibility and architecture choices. They are **not measurements of real photonic hardware latency, energy efficiency, throughput or optical behaviour**. Those questions require later experiments on actual Q.ANT hardware.

## Research direction

The work began with conventional MNIST architecture searches and progressively moved toward Q.ANT-specific operations.

The current line of research studies Q.ANT's Fourier/KAN-style operation exposed through `calc_kan_layer_fprop`. A recurring observation is that networks trained with an exact cosine reference can diverge numerically when evaluated through the Q.ANT CPU backend, especially as higher Fourier frequencies are introduced.

Recent experiments have therefore moved from simply measuring accuracy to investigating the mechanism:

- frequency-dependent reference-to-Q.ANT numerical error;
- separation of bfloat16 effects from the remaining Q.ANT-specific residual;
- accumulated error at the model logits;
- interaction between logit perturbation and classification decision margins;
- architecture interventions intended to retain model capacity while reducing deployment mismatch.

**Experiment 018** was a historical intervention. The original Toolkit discovery series is preserved through **Exp030** (including the AST-equivalent readability-only refactor from #152). **PNN-v1** is a separately numbered architecture series with committed Exp001–Exp021 results and auxiliary paired confirmations at this snapshot. See [the current protocol map](docs/current-systems-and-protocols.md); do not treat a historical result as a currently authorized future protocol.

## AI-assisted research and approval boundaries

**AI Researcher v2** is the connected, **Groq-backed** proposal/compiler/review path: its manual or authorized feedback-triggered workflow invokes `scripts.ai_researcher_v2`, requires `GROQ_API_KEY`, and makes real calls to Groq. The `deterministic` / `allow_paid_usage=false` settings in `research_queue/researcher_backend.json` belong to a separate non-wired prototype and **do not disable v2 API usage or guarantee zero provider cost**. **AI Researcher v3** has separate offline, read-only strategy-isolated Research Memory and manually approved bounded live Groq sessions. Its proposals, reviews and unapproved standalone drafts are not completed experiment results.

```text
Versioned results -> proposal + compiler checks
 -> human scientific review of exact proposal SHA
 -> separate human compute authorization
 -> immutable approved job + pinned execution commit
 -> guarded Q.ANT CPU/software-simulation run
 -> result + execution record published via PR
```

Older direct-approval workflows are retained but disabled. **Dev-4 selection evidence remains frozen and strategy-isolated**; it must not be silently merged into general agent memory. Entering either v3 live-workflow approval phrase **does authorize that bounded Groq API session**, possibly with multiple model calls, but never Q.ANT training. V2 workflow dispatch can likewise initiate external Groq usage. Code merge, an AI proposal and a Groq session are not human compute authorization. Consult the [current systems and protocol map](docs/current-systems-and-protocols.md) for exact entrypoints and approval scopes.

## Experimental progression

The repository preserves the complete progression rather than only successful experiments. That includes hypotheses that were not supported.

Broadly, the programme has progressed through:

1. MNIST baselines and width/depth searches.
2. Automated topology search and Pareto analysis.
3. Multi-seed and full-dataset validation.
4. Cross-dataset validation on FashionMNIST.
5. Q.ANT-native Fourier/KAN architecture experiments.
6. Fourier-capacity and training-robustness tests.
7. Numerical mismatch diagnostics and error decomposition.
8. Activation-error occupancy and logit-margin diagnostics.
9. Frequency-controlled architecture intervention.
10. Accumulation, low-fan-in and predictive compatibility studies through original Toolkit **Exp030**.
11. The separately numbered **PNN-v1** ECG200/architecture work, with committed **Exp001–Exp021**, plus distinct confirmation records.

Negative results are kept because they constrain the next hypothesis. For example, adding output noise during training did not materially reduce the Q.ANT/reference disagreement, and a first Q.ANT-fitted training surrogate did not satisfy its preregistered criterion across both g4 and g8.

## Repository structure

```text
experiments/                 Historical Toolkit discovery Exp001–Exp030
results/                     Published discovery and auxiliary results
pnn-v1/                      Separate PNN proposal, experiment, result, analysis,
                             model, dataset and benchmark namespaces
research_queue/
  proposals/                 Historical proposal lineage
  ai_researcher/             Versioned v2 AI proposals
  human_review/              Scientific decisions/history
  compiled/                  Compiler checks, not compute authorization
  jobs/                      Guarded jobs
  benchmarks/                Frozen strategy-isolated Dev-4 records
  v3_hypotheses/             Source-pinned but untested v3 proposals
  v3_protocols/              Unapproved standalone v3 draft
docs/current-systems-and-protocols.md
                             Maintained status/protocol map
public/research-dashboard.json
                             Derived public dashboard, not raw evidence
public/experiment-catalog.json
                             Derived, mirror-deduplicated discovery snapshot
.github/workflows/           Current, manual-only, disabled legacy, validation
```

Identifiers are scoped to their experiment family. Some PNN output is mirrored into root `results/`; a future global catalog must reconcile identities and provenance instead of counting mirrors as independent experiments.

## Global experiment discovery

A [global experiment catalog](docs/global-experiment-catalog.md) now indexes committed Toolkit/PNN-v1 results, proposal documents and explicitly unexecuted v3 planning records. It uses composite identities, source hashes, per-record status and source/recorded-execution pointers, and deduplicates byte-identical PNN mirrors in root `results/`. The [committed JSON snapshot](public/experiment-catalog.json) is **derived**; regenerate or freshness-check it after new results. It is for human/audit discovery, **not** a new Dev-4/v3 model memory source, a general experiment ranking, or a means of bypassing human compute approval.

## Reproducibility and compute

The experiments use fixed seeds where appropriate and store machine-readable JSON results. Historical Toolkit discovery used MNIST/FashionMNIST; separately numbered PNN-v1 and the frozen selection benchmark use ECG200 with their own paired-control protocols. Seed lists, thresholds and permissions must be scoped to each study.

The automation enforces declared limits such as maximum parameter count, number of runs, epochs and dataset size. This is designed so that inexpensive experiments can identify promising directions before more expensive validation is attempted.

The longer-term compute strategy is:

> **cheap experiments → identify promising candidates → rigorous validation → scale promising experiments**

GitHub Actions is suitable for the current CPU-scale research. A later goal is to make promising experiments portable to larger research infrastructure such as HPC systems, and ultimately to validate hardware-specific hypotheses on real photonic hardware.

## Why this may matter

Alternative accelerators raise a broader research question: should neural networks continue to be designed primarily around GPU-friendly operations, or can architectures be co-designed with the primitives offered by emerging hardware?

This project treats the architecture itself as an experimental variable. The potential value is not limited to one model or benchmark. The methodology may be useful for:

- Q.ANT and developers investigating software/hardware co-design;
- researchers in photonic and alternative computing;
- hardware-aware neural architecture search;
- European HPC and AI research infrastructure;
- open and reproducible AI research.

At this stage these are research motivations, **not claims that the project has demonstrated superior photonic performance**.

## Relationship to Q.ANT

This is an **independent Debatt-AI research project** using Q.ANT's open-source Native Computing Toolkit. It is not presented as research performed, sponsored or endorsed by Q.ANT.

Q.ANT's toolkit is the technical foundation for the Q.ANT-specific experiments, while the hypotheses, automation, experiment design and interpretations in this repository belong to the Debatt-AI Q.ANT Research Lab.

## Public dashboard

A public-facing summary of the research is available on Debatt-AI:

https://www.debatt-ai.se/qant

The dashboard is generated from version-controlled result files in this repository so that the public presentation remains connected to the underlying experiments.

## Research principles

- **Falsifiable hypotheses before compute.**
- **Reproducible results rather than isolated demos.**
- **Negative results are useful results.**
- **Human approval before autonomous compute.**
- **No photonic performance claims from CPU-backend measurements.**
- **Scale compute only when earlier evidence justifies it.**

## License and upstream project

This repository contains Debatt-AI's research code and results. Q.ANT's Native Computing Toolkit is a separate upstream open-source project; consult its repository and license for the toolkit's own terms.

---

**Status (29 September 2026):** Toolkit discovery retained through Exp030; separate PNN-v1 results through Exp021; v2 guarded research orchestration; v3 research prototypes; frozen Dev-4 benchmark. Consult the [current protocol map](docs/current-systems-and-protocols.md) for source-of-truth links.
