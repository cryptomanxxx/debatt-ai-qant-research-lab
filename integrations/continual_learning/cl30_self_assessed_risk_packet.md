# CL-30: bounded self-assessed risk decision packet (NO_GO)

Status: **REVIEW ONLY / NO_GO**. This document is not an execution authorization, an independent audit, or a claim that model inference has happened. It does not modify CL-24, CL-28, or CL-29 safety gates.

## Exact proposed future scope

- GitHub-hosted **standard** `ubuntu-latest` runner for a public repository; no paid/larger runner, cloud VM, self-hosted runner or special GPU.
- Source-locked, fixed-input container; seed **101**, frozen T1 **train** item index **0** only, no held-out data, no network.
- Exactly three model variants: `local_hybrid`, `sparse_local_hybrid`, `sparse_shared_index`; at most **three forward evaluations total**.
- No training, loss/gradient evaluation, optimizer or weight update, dataset sweep, energy measurement, or unapproved output.
- First probe assesses **numerical parity only**. Shared-index reuse is a training-step optimization and is **not** exercised by standalone forward/predict, so no compute-efficiency or continual-learning claim follows.
- All observations must be labeled as actual results and separated from synthetic fixtures; output bound at most 1 MiB. No actual observations exist.

## Trust boundary, evidence, and residual risks

| Property | Existing evidence or provider documentation | Residual limitation | Decision |
| --- | --- | --- | --- |
| Host runner | Standard GitHub-hosted runners are provisioned as temporary VMs (GitHub Docs: https://docs.github.com/en/actions/reference/runners/github-hosted-runners). | GitHub provider assertion; this project has **not independently verified** host hypervisor/kernel isolation. | **UNRESOLVED**, not independently signed off |
| Container outbound network | CL-25/26/27 invoke `docker run --network=none`; probes rejected IPv4 TCP/UDP and IPv6 TCP or found it unavailable. Docker none driver: https://docs.docker.com/engine/network/drivers/none/ . | Loopback remains; these probes do not prove absence of every possible covert channel or host-side build-step networking. | Runtime network gate evidenced, not blanket security certification |
| Container privileges | CL-26/27 verified `--cap-drop=ALL`, `--security-opt=no-new-privileges`, `--read-only`, `--user=65534:65534`, 32 PIDs, 128 MiB, 1 CPU. | Containers share the Linux kernel of their GitHub runner VM; malicious code or kernel bugs are outside observed probes. | Restriction evidenced, residual kernel risk |
| Repository credentials | Existing CL-25/26/27 workflows set `permissions: contents: read`; runtime container has no proposed secret mounts. | The **host job** may still receive a read-only token, and checkout/build steps have network access before `docker run`. A future workflow must not forward credentials, the Docker socket, or host mounts to model code. | Must be verified in final execution workflow |
| Image provenance | CL-22 source files pinned; CL-25 sealed one input; CL-27 repeatable local image configuration ID. | Local Docker image ID is **not** an independently signed registry OCI manifest; final image+runner identity and provenance are not approved. | **UNRESOLVED** |
| Mathematical comparison | CL-28 proposes `math.isclose` with absolute 1e-12, relative 1e-10; CL-29 limits interpretation. | Tolerances have synthetic fixture tests but no independently approved numerical error analysis or real-model observation. | **UNRESOLVED**; no claim of scientifically established tolerance |
| Cost | Standard GitHub-hosted runners for public repositories are free (https://docs.github.com/en/actions/concepts/billing-and-usage). | This does not authorize paid runner selections or unlimited use; GitHub terms/limits still apply. | Use standard public-repo runner only |

## Owner-controlled risk decision, not academic credentialing

A professor or university affiliation is **not required** merely to design and perform independent research. The owner may ultimately decide that the documented VM/provider boundary and small trusted first-party code scope are an acceptable residual risk. That is an **owner risk-acceptance decision**, not an independent security certification, and must be recorded explicitly. Code-review bots are supplementary review, not a substitute for independent attestation.

This packet **does not make that decision** and cannot turn unresolved flags to true. If a future exact scope requires independent audit, obtain it or change that requirement **only with the owner's explicit decision** and accurate description of the resulting assurance level.

## Required separate future approvals

1. Review final trusted model image, sample, runner and workflow identities; ensure fail-closed checks and no host secret exposure.
2. Review numerical tolerance rationale. If not independently validated, label results *exploratory parity within candidate tolerance*.
3. Record owner's explicit acceptance or rejection of the documented residual risks, including lack of independent host assessment; do not mislabel this as third-party signoff.
4. Obtain Marcus's separate **exact inference authorization** identifying run and boundaries. This is **not** an authorization for training, energy measurement or any wider probe.
5. Only in a later, separately reviewed change may an inference workflow be added. The existing NO_GO guard remains intact meanwhile.

**Current decision: NO_GO. No model imported, inference performed, training started, or physical energy measured for CL-30.**
