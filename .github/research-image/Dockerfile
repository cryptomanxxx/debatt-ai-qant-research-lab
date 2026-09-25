FROM python:3.13-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    LIBCLANG_PATH=/usr/lib/llvm-18/lib

RUN apt-get update \
 && apt-get install -y --no-install-recommends git curl clang libclang-dev build-essential ca-certificates cargo rustc \
 && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip \
 && python -m pip install --index-url https://download.pytorch.org/whl/cpu torch \
 && python -m pip install "numpy==2.3.5" "ml_dtypes==0.5.4" "aeon==1.6.0" "maturin>=1.14,<2.0"

RUN git clone --depth 1 --recurse-submodules https://github.com/Q-ANT-GmbH/qant_native_computing_toolkit.git /tmp/qant \
 && cd /tmp/qant \
 && maturin build --release -F cpu-backend \
 && python -m pip install target/wheels/*.whl \
 && rm -rf /tmp/qant

RUN python -c "import torch, numpy, ml_dtypes, aeon, qant_native_computing_toolkit; print('research environment ready')"

WORKDIR /workspace
