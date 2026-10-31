"""Read-only availability probe for optional CL-1 simulation toolchains.

This deliberately does not import or execute any simulator, install dependencies,
start model training, or make hardware-performance claims.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path


def available(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, AttributeError, ValueError, ModuleNotFoundError):
        return False


def inspect_backends(simphony_root: str | None = None) -> dict:
    root = Path(simphony_root).expanduser().resolve() if simphony_root else None
    simphony_source = bool(root and (root / "onnarchsim" / "simulator.py").is_file())
    # SimPhony/onnarchsim is a SOURCE checkout, not BYU's unrelated 'simphony' PyPI package.
    return {
        "qant_cpu": {
            "available": available("qant_native_computing_toolkit"),
            "evidence": "module discovery only; no backend invocation",
        },
        "lava_cpu": {
            "available": available("lava"),
            "evidence": "module discovery only; Loihi 2 hardware is not tested",
        },
        "simphony_source": {
            "available": simphony_source,
            "path": str(root) if root else None,
            "evidence": "verified onnarchsim/simulator.py exists; runtime and CACTI not validated",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--simphony-root",
        default=os.environ.get("DEBATT_AI_SIMPHONY_ROOT"),
        help="Path to a separate ScopeX-ASU/SimPhony checkout",
    )
    parser.add_argument("--strict", action="store_true", help="Fail if any optional backend is missing")
    args = parser.parse_args()
    report = inspect_backends(args.simphony_root)
    print(json.dumps({"type": "availability_only", "backends": report}, indent=2, sort_keys=True))
    return int(args.strict and not all(x["available"] for x in report.values()))


if __name__ == "__main__":
    raise SystemExit(main())
