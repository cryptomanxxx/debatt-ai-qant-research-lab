# CL-36 deterministic 21-file candidate. No model import or inference in CI.
# CL-34 20-file audited context + dormant one-shot runner .github/cl36/once.py.
# Default command is HARD NO_GO. The --approved-once command is NOT used by CI.
# Final Docker config ID must be rechecked; this is not a signed OCI manifest.
FROM python:3.11-slim@sha256:673719e071ecf27434fd2635bbd8d00f49f4fdd81f243143438a42ced8bedf60
COPY . /
USER 65534:65534
ENTRYPOINT ["python3", "-I", "-B", "/opt/research/.github/cl36/once.py"]
