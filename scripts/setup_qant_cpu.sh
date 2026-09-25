#!/usr/bin/env bash
set -euo pipefail

CACHE_DIR="${QANT_WHEEL_CACHE_DIR:-}"
if [ -n "$CACHE_DIR" ]; then
  mkdir -p "$CACHE_DIR"
  cached_wheel=( "$CACHE_DIR"/qant_native_computing_toolkit-*.whl )
  if [ -e "${cached_wheel[0]}" ]; then
    echo "Using cached Q.ANT CPU wheel: ${cached_wheel[0]}"
    python -m pip install "${cached_wheel[0]}"
    python -c "import qant_native_computing_toolkit as q; print(q)"
    exit 0
  fi
fi

if command -v sudo >/dev/null 2>&1; then
  sudo apt-get update -qq
  sudo apt-get install -y clang libclang-dev
else
  apt-get update -qq
  apt-get install -y clang libclang-dev
fi

if ! command -v cargo >/dev/null 2>&1; then
  curl https://sh.rustup.rs -sSf | sh -s -- -y
fi
export PATH="$HOME/.cargo/bin:$PATH"

python -m pip install "maturin>=1.14,<2.0"

if [ ! -d qant_native_computing_toolkit ]; then
  git clone --recurse-submodules https://github.com/Q-ANT-GmbH/qant_native_computing_toolkit.git
fi

cd qant_native_computing_toolkit
export LIBCLANG_PATH="${LIBCLANG_PATH:-/usr/lib/llvm-18/lib}"
maturin build --release -F cpu-backend
if [ -n "$CACHE_DIR" ]; then
  cp -f target/wheels/*.whl "$CACHE_DIR"/
fi
python -m pip install target/wheels/*.whl
python -c "import qant_native_computing_toolkit as q; print(q)"
