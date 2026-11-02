"""CL-34: stage exact 20-file epoch-zero first-forward NO_GO image.

Extends CL-31's verified 19-file boundary context with one guarded CL-33
candidate script. Staging, builds and probes never import any model.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "integrations" / "continual_learning"))
from cl31_full_boundary_build_context import stage_full_boundary
from cl27_minimal_build_context import EXPECTED

CL26 = "integrations/continual_learning/cl26_linux_boundary_probe.py"
CL33 = "integrations/continual_learning/cl33_first_forward_candidate.py"


def stage_candidate(destination: Path) -> dict:
    result = stage_full_boundary(destination)
    destination = destination.resolve()
    source = ROOT / CL33
    target = destination / "opt" / "research" / CL33
    if (not source.is_file() or source.is_symlink()
            or not source.resolve().is_relative_to(ROOT)
            or target.exists() or not target.parent.is_dir()
            or CL33 in EXPECTED or CL33 == CL26):
        raise ValueError("CL-34 NO_GO: invalid guarded candidate input")
    shutil.copyfile(source, target)
    target.chmod(0o644)
    os.utime(target, (0, 0))
    for item in sorted(destination.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        if item.is_dir():
            item.chmod(0o755)
            os.utime(item, (0, 0))
    os.utime(destination, (0, 0))
    files = sorted(item.relative_to(destination / "opt" / "research").as_posix()
                   for item in destination.rglob("*") if item.is_file())
    if files != sorted((*EXPECTED, CL26, CL33)):
        raise ValueError("CL-34 NO_GO: unexpected file in sealed image")
    return {**result, "staged_files": 20,
            "guarded_candidate": CL33, "decision": "NO_GO",
            "models_imported": 0, "forward_calls_executed": 0}


if __name__ == "__main__":
    print(json.dumps(stage_candidate(Path(sys.argv[1])), sort_keys=True))
