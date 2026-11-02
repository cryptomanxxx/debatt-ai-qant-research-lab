"""Stage CL-31 deterministic 19-file NO-MODEL container context.

Reuses CL-27's 18-file audited allowlist, then adds only the CL-26
boundary probe. Does not import models or execute any model code.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import sys

from cl27_minimal_build_context import ROOT, EXPECTED, stage

EXTRA = "integrations/continual_learning/cl26_linux_boundary_probe.py"


def stage_full_boundary(destination: Path) -> dict:
    result = stage(destination)
    destination = destination.resolve()
    source = ROOT / EXTRA
    if (not source.is_file() or source.is_symlink()
            or not source.resolve().is_relative_to(ROOT)
            or EXTRA in EXPECTED):
        raise ValueError("CL-31 NO_GO: invalid extra boundary probe")
    target = destination / "opt" / "research" / EXTRA
    if target.exists() or not target.parent.is_dir():
        raise ValueError("CL-31 NO_GO: unexpected staged destination")
    shutil.copyfile(source, target)
    target.chmod(0o644)
    os.utime(target, (0, 0))
    for item in sorted(destination.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        if item.is_dir():
            item.chmod(0o755)
            os.utime(item, (0, 0))
    os.utime(destination, (0, 0))
    files = sorted(p.relative_to(destination / "opt" / "research").as_posix()
                   for p in destination.rglob("*") if p.is_file())
    if files != sorted((*EXPECTED, EXTRA)):
        raise ValueError("CL-31 NO_GO: staged file allowlist drift")
    return {**result, "staged_files": 19,
            "additional_boundary_probe": EXTRA, "decision": "NO_GO"}


if __name__ == "__main__":
    print(json.dumps(stage_full_boundary(Path(sys.argv[1])), sort_keys=True))
