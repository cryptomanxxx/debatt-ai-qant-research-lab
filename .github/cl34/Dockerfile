# CL-34 exact 20-file sealed first-forward candidate (review only, HARD NO_GO).
# CL-31 19 audited files + one CL-33 guarded candidate, no held-out data.
# Builds never import models; default command performs only zero-execution preflight.
# A local Docker configuration ID is NOT a signed OCI registry manifest.
FROM python:3.11-slim@sha256:673719e071ecf27434fd2635bbd8d00f49f4fdd81f243143438a42ced8bedf60
COPY . /
USER 65534:65534
ENTRYPOINT ["python3", "-I", "-B", "/opt/research/integrations/continual_learning/cl33_first_forward_candidate.py"]
