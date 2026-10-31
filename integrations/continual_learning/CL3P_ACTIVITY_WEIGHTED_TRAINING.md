# CL-3P: proposed training-step cost, weighted by input activity

**NO_GO: no model training or actual updates are performed.**

CL-3E previously projected a fixed number of training operations under
a constant assumed activity level. CL-3P instead sums those CL-2 source-loop
estimates over the **predeclared synthetic training input features**.
Each input is assigned one *hypothetical* update, respecting the proposal's
128 examples per task × four tasks = 512 updates per arm/seed.

The two primary comparison arms are dense SGD and local hybrid. Sparse and
shared-index variants are **engineering ablations** and must not be confused
with independent learning-rule controls.

This counts logical parameter reads/writes, source-level forward weight
terms, feature checks and payload-byte proxies using assumed compact widths
(8-byte parameters, 1-byte inputs, 2-byte active indices). It explicitly
does **not** include actual model execution, optimizer behavior over time,
measured memory traffic, Python object allocation, CPU cycles, energy,
accuracy or photonic/Loihi hardware behavior.

CL-3O separately models hypothetical heldout inference operations using
the distribution of **features**, not labels. CL-3P examines *training*
features only and never accesses heldout labels. No conclusion about
energy savings or useful learning follows without approved experiments.
CL-3A remains NO_GO pending scientific review and Marcus's separate
human compute authorization.
