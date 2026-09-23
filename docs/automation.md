# Research automation

The lab is moving from interactive Colab runs toward a runner-independent
research pipeline.

## Phase 1: automatic result publishing

The GitHub Actions workflow in `.github/workflows/qant-research.yml` can be
started manually from the Actions tab. It checks out the repository, installs
the Q.ANT CPU backend, runs the supplied experiment command, copies generated
files from `local_results/` into `results/`, commits them, and pushes them
back to the repository.

The workflow uses GitHub's temporary `GITHUB_TOKEN`; no personal access token
is stored in the repository. Its explicit permission is limited to repository
contents write access.

## Runner-independent contract

Experiments should:
1. be runnable from one command;
2. write machine-readable outputs to `local_results/`;
3. not contain runner-specific authentication;
4. record reproducibility metadata in their result JSON.

This contract lets the execution backend later move from GitHub-hosted runners
to another cloud runner, Sweden AI Factory, EuroHPC, or photonic hardware
without redesigning the experiment code.

## Security

Do not commit personal access tokens, cloud credentials, or other secrets.
Keep workflow permissions at the minimum required level. Self-hosted runners
need additional isolation and trust controls, especially for public
repositories.
