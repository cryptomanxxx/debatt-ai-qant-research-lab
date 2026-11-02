"""Stage a 21-file CL-36 candidate from the CL-34 locked NO_GO context.

Adds only the separately reviewable, dormant one-shot runner source.
Never imports model code or enters the approved-once code path.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / ".github" / "cl34"))
from stage_context import stage_candidate
sys.path.insert(0, str(ROOT / "integrations" / "continual_learning"))
from cl27_minimal_build_context import EXPECTED

CL26 = "integrations/continual_learning/cl26_linux_boundary_probe.py"
CL33 = "integrations/continual_learning/cl33_first_forward_candidate.py"
CL36 = ".github/cl36/once.py"


def stage_execution_candidate(destination: Path) -> dict:
    result = stage_candidate(destination)
    destination = destination.resolve()
    source = ROOT / CL36
    target = destination / "opt" / "research" / CL36
    if (not source.is_file() or source.is_symlink()
            or not source.resolve().is_relative_to(ROOT)
            or target.exists() or not target.parent.is_dir()):
        raise ValueError("CL-36 NO_GO: invalid candidate runner path")
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
    if files != sorted((*EXPECTED, CL26, CL33, CL36)):
        raise ValueError("CL-36 NO_GO: stage allowlist drift")
    return {**result, "decision": "NO_GO", "staged_files": 21,
            "dormant_runner": CL36, "models_imported": 0,
            "forward_calls_executed": 0}


if __name__ == "__main__":
    print(json.dumps(stage_execution_candidate(Path(sys.argv[1])), sort_keys=True))
