# CL-8J — Google Cloud feasibility and safety checklist (NO PROVISIONING / NO MODEL)

**NO_GO / design-only.** Screened 2026-11-01. This document does not create an account, cloud project, VM, firewall, secret, cost commitment, or new network test.

## Why Google Cloud is a candidate, not yet approved

Google Cloud advertises a **conditional e2-micro Free Tier**, not an unconditional promise of zero cost. Eligibility depends on the supported regions, monthly instance-hours and other resource quotas. Disks, traffic, IP addresses and additional services must be priced separately before provisioning. Consult the [official Free Program terms](https://docs.cloud.google.com/free/docs/free-cloud-features).

Google VPC has **implied allow egress**. A secure experiment design must explicitly add externally enforced deny rules that cover applicable IPv4 **and** IPv6 paths and verify rule precedence. See [Google firewall documentation](https://docs.cloud.google.com/firewall/docs/firewalls). **No rules have been configured**.

Google [billing budgets](https://docs.cloud.google.com/billing/docs/how-to/budgets) used only for alerts do **not** cap spending. [Spend caps](https://docs.cloud.google.com/billing/docs/how-to/budgets-spend-caps) may be available for certain eligible services/accounts, but we have **not** established applicability to an e2-micro VM or related charges. We must not assume a hard zero-cost limit.

Google warns that processes running on a VM with an attached service account can obtain tokens from the metadata server. Review [service-account hardening](https://docs.cloud.google.com/iam/docs/best-practices-service-accounts) and ensure that future experimental processes have no unnecessary ambient credentials.

## Proposed architecture — documentation only

1. Privately administered, dedicated cloud project, **not** exposed to untrusted public-repository PR code.
2. Verified eligible e2-micro configuration, storage and ingress/egress pricing; no VM or resource provisioning without separate permission.
3. Externally enforced VPC egress deny plus minimum necessary host firewall controls. Test for IPv4, IPv6 and unexpected metadata/API access. The experiment process must not be able to change policy.
4. Externally enforced CPU, memory, process, wall, disk and output limits. VM instance size is not a per-process quota.
5. A reviewed *fixed non-model* diagnostic to test denied egress to a controlled, authorized destination and log the policy decision; a namespace inode change is not sufficient evidence.
6. Immutable reviewed code/input provenance, deterministic results and independent threat review.
7. Only after separate scientific GO and Marcus's explicit workload-specific approval can any AI training/inference be considered.

## Evidence gaps and hard stops

The [machine-readable design checklist](cl8j_gcp_feasibility.json) leaves every gate false. A successful CI check is not an independent security review. This design does not assert that VPC isolation alone protects the host, that a free-tier quota caps all spending, or that external egress denial has been tested.

Next *human decision*, if CL-8J is accepted: whether to investigate eligibility manually, without creating paid resources. Account registration, billing setup, provisioning and any test traffic require separate authorization and an exact cost/safety plan.

[CL-8H host acceptance](cl8h_alternative_host_security_contract.json) and [CL-8C execution envelope](cl8c_cpu_execution_envelope.json) remain **NO_GO**.

**No AI model, training, inference, new data, network probe, cloud VM or energy measurement.**
