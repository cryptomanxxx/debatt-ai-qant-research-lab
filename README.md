# Debatt-AI Q.ANT Research Lab

An open, experimental research project exploring **neural-network architectures for Q.ANT's Native Computing Toolkit** and, ultimately, photonic computing.

The central question is not simply:

> How can an existing neural network be executed faster on new hardware?

It is:

> **What should neural networks look like when they are designed around the computational primitives of a different kind of processor?**

This repository contains the experiments, results and research automation behind the public Debatt-AI Q.ANT Research Lab.

## Current status

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

**Experiment 018** is the current intervention. It compares a four-component Fourier model, a standard eight-component model using frequencies 1–8, and an eight-component frequency-controlled model that repeats frequencies 1–4. The goal is to test whether the higher frequencies themselves drive the larger deployment mismatch while holding the number of trainable components constant.

## Autonomous research loop

The repository is also an experiment in **AI-assisted scientific research**.

The current workflow is:

```text
Experiment results
      ↓
GPT-5.6 Sol research analysis
      ↓
Falsifiable hypothesis
      ↓
Human review and approval
      ↓
Guarded GitHub Actions experiment
      ↓
Version-controlled results
      ↓
Next analysis
```

The AI Researcher can analyse completed experiments, formulate a falsifiable hypothesis, design the next experiment and prepare its implementation. Compute is deliberately behind a human approval boundary: a proposed experiment does not execute until its declared budget has been reviewed and approved.

This separation is intentional. It makes the research loop increasingly autonomous without giving an AI system unrestricted compute execution.

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

Negative results are kept because they constrain the next hypothesis. For example, adding output noise during training did not materially reduce the Q.ANT/reference disagreement, and a first Q.ANT-fitted training surrogate did not satisfy its preregistered criterion across both g4 and g8.

## Repository structure

```text
experiments/                 Reproducible experiment implementations
results/                     Version-controlled experiment results
research_queue/
  proposals/                 AI Researcher hypotheses and proposed methods
  analyses/                  Formal analyses of completed experiments
  jobs/                      Human-approved compute jobs
scripts/                     Setup, orchestration, progress and dashboard tools
public/research-dashboard.json
                             Stable public research summary
.github/workflows/           Guarded GitHub Actions automation
```

Each experiment is intended to leave a reproducible trail from **observation → hypothesis → approval → execution → result → analysis**.

## Reproducibility and compute

The experiments use fixed seeds where appropriate and store machine-readable JSON results in the repository. Current larger validation runs use FashionMNIST with multiple seeds, while earlier experiments used MNIST.

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

**Status:** Active research. Current frontier: Q.ANT-specific Fourier/KAN numerical compatibility and architecture design.
