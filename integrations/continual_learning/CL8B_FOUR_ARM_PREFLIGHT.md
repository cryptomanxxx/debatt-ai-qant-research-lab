# CL-8B — Four-arm bounded workload preflight (NO TRAINING)

**Status: NO_GO / READ-ONLY PROJECTION.** CL-3B's earlier preflight describes a two-arm CL-2 draft. CL-8B calculates separate **proposed workload ceilings** from the more recent CL-6A four-arm / five-seed study. It does not import or instantiate AI models, load or generate synthetic samples, infer, train, time the CPU, or read a power meter.

## Proposed upper bounds, not executed counts

| Proposed experiment axis | Value |
| --- | ---: |
| Architectures | 4 |
| Paired seeds | 5 (101–105) |
| Sequential distributions | 4 |
| Checkpoints per seed and arm | 5 |
| Proposed maximum updates per seed/arm | 512 |
| **Maximum planned updates overall** | **10,240** |
| **Maximum planned heldout predictions overall** | **25,600** |

The counts are algebraic ceilings: `5 seeds × 4 arms × 512 updates` and `5 seeds × 4 arms × 5 checkpoints × 4 tasks × 64 heldout records`. They are not evidence of work performed, classification quality, physical memory traffic, seconds, or joules. No training compute budget or runtime allowance is approved by these counts.

## GitHub Actions on a phone or tablet

1. Open **Actions → CL-8B Four-Arm Workload Preflight (NO TRAINING)**.
2. Select **Run workflow** when you want the static preflight to display.
3. After it finishes, open the run's **Summary** to see the four-arm workload table and explicit **NO_GO** state.

The workflow is `workflow_dispatch` only and uses read-only repository permissions. Its sole Python operations are a static audit, no-compute tests, and writing the GitHub **job summary**. It does **not** contain a training or data-generation job or an `--execute` mode.

For a manual source-only check, run:

```bash
python -m unittest -v tests.test_cl8b_four_arm_preflight
python -m integrations.continual_learning.cl8b_four_arm_preflight --preflight
```

## Scientific stop conditions

The preflight validates CL-6A's unapproved protocol and the CL-7F nine-gate NO_GO review. It rejects changes to the preregistered shape, arm/seed/task counts, update/sample bounds, or false training authorization. CL-7A's six blocking findings remain OPEN. The CL-8A review packet has not been sent to a researcher; an external review is still pending.

This produces an understandable readiness dashboard, **not** an executable scientific experiment. Dataset generation, model inference, model training, CPU timing and calibrated electrical-energy tests each require separately reviewed protocols and, where applicable, Marcus's explicit authorization.
