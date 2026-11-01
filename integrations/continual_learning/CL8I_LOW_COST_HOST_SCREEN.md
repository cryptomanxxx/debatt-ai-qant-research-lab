# CL-8I — Preliminary low-cost execution host screen (NO MODEL / NO TRAINING)

**Status: REVIEW ONLY / NO_GO.** Screened on 2026-11-01 from provider documentation. This is not a benchmark, a cost guarantee, host provisioning, or a security approval. Pricing and free-tier eligibility can change.

## Candidates

| Environment | Cost and availability caveat | Control / outstanding barrier | Decision |
| --- | --- | --- | --- |
| Current GitHub-hosted Ubuntu | Existing CI allowance depends on account usage | The [fixed child run](https://github.com/cryptomanxxx/debatt-ai-qant-research-lab/actions/runs/41343841231) received `OS_PERMISSION_DENIED`. It is useful for static CI, but the tested namespace isolation cannot be assumed. | NO_GO |
| Google Cloud Compute Engine e2-micro | Conditional free-tier usage for eligible regions/configurations; billing, storage and egress restrictions apply | Account administrator can specify **VPC egress-deny firewall rules**, but default egress is allowed and **we have configured nothing**. Candidate for *fixed non-model* validation only. | NO_GO |
| Oracle Cloud Always Free VM | Eligibility, usage quotas and regional capacity availability apply | VM/network configuration and externally enforced denial remain untested. | NO_GO |
| An already-owned computer under private administrative control | No hardware purchase; electricity and operational costs remain | Could be a manually reviewed isolated host, but publicly triggered self-hosted GitHub Actions runners are a serious security hazard. | NO_GO |

### Primary provider references

- [Google Cloud Compute Engine free-tier overview](https://cloud.google.com/products/compute) and [Free Program terms](https://docs.cloud.google.com/free/docs/free-cloud-features).
- [Google Cloud firewall rules](https://docs.cloud.google.com/firewall/docs/firewalls): virtual networks implicitly allow egress absent higher-priority deny, so an explicit external policy is needed.
- [Oracle Always Free resources and capacity caveats](https://docs.oracle.com/en-us/iaas/Content/FreeTier/freetier_topic-Always_Free_Resources.htm).
- [GitHub Actions secure use](https://docs.github.com/en/actions/reference/security/secure-use): GitHub warns against self-hosted runners in public repositories.

## Preliminary recommendation (NOT a selection)

1. Evaluate **Google Cloud e2-micro** documentation and account eligibility first, because a documented VPC-layer egress-deny configuration is possible; this is not a claim it will satisfy all CL-8H gates.
2. Consider **Oracle Always Free** if capacity, availability and independent network enforcement can be verified.
3. A private existing computer remains an option for local validation only if the operator controls isolation, reviews the code and avoids untrusted public-PR-triggered self-hosted runner execution.

Do not create an account, attach payment details, configure a firewall, reserve capacity, or spend money without Marcus's separate consent. Before any cloud test, agree on a *fixed non-AI outbound-denial verification procedure*, a controlled destination, and hard cost ceilings. No actual outbound probe is included.

Each candidate remains **NO_GO**, all [CL-8H gates](cl8h_alternative_host_security_contract.json) remain unverified, and [CL-8C's general runner](cl8c_cpu_execution_envelope.json) remains unimplemented. Our [machine-readable comparison](cl8i_low_cost_host_screen.json) contains no measured costs, energy, AI throughput or isolation performance.

**No model, training, inference, data generation, host provisioning, network probe or energy measurement.**
