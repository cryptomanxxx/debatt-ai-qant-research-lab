"""Build an epoch-normalized minimal Docker context, without model imports.

Only the explicit CL-25 Dockerfile COPY allowlist may enter the temporary
context. Outputs exist only in GitHub Actions RUNNER_TEMP, never the repo.
"""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
EXPECTED = (
    "integrations/__init__.py",
    "integrations/continual_learning/cl1_cpu.py",
    "integrations/continual_learning/cl2_mlp_baseline.py",
    "integrations/continual_learning/cl2_local_mlp.py",
    "integrations/continual_learning/cl2_sparse_forward.py",
    "integrations/continual_learning/cl2_shared_index_forward.py",
    "integrations/continual_learning/cl10j_source_event_collector.py",
    "integrations/continual_learning/cl10k_weight_update_hooks.py",
    "integrations/continual_learning/cl10l_forward_hooks.py",
    "integrations/continual_learning/cl10m_training_hooks.py",
    "integrations/continual_learning/cl22_model_source_lock.py",
    "integrations/continual_learning/cl22_model_source_lock.json",
    "integrations/continual_learning/cl20_no_model_container_probe.py",
    "integrations/continual_learning/cl19_single_sample_forward_candidate.json",
    "integrations/continual_learning/cl23_no_model_source_container_probe.py",
    "integrations/continual_learning/cl25_single_archived_input.json",
    "integrations/continual_learning/cl25_single_input_audit.py",
    "integrations/continual_learning/cl25_no_model_single_input_probe.py",
)


def stage(destination: Path) -> dict:
    runner_temp = Path(os.environ["RUNNER_TEMP"]).resolve()
    destination = destination.resolve()
    if destination.parent != runner_temp or destination.exists():
        raise ValueError("CL-27: staging path must be new and directly inside RUNNER_TEMP")
    dockerfile = (ROOT / "integrations/continual_learning/Dockerfile.cl25-one-input-probe").read_text()
    active = [
        line.split()[1] for line in dockerfile.splitlines()
        if line.startswith("COPY ")
    ]
    if tuple(active) != EXPECTED:
        raise ValueError("CL-27: staging allowlist diverges from reviewed CL-25 Dockerfile")
    destination.mkdir(mode=0o755)
    for name in EXPECTED:
        source = (ROOT / name).resolve()
        if not source.is_file() or source.is_symlink() or not source.is_relative_to(ROOT):
            raise ValueError("CL-27: missing/nonregular input: " + name)
        target = destination / "opt" / "research" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        target.chmod(0o644)
        os.utime(target, (0, 0))
    directories = sorted(
        [destination, *destination.rglob("*")],
        key=lambda item: len(item.parts), reverse=True,
    )
    for directory in directories:
        if directory.is_dir():
            directory.chmod(0o755)
            os.utime(directory, (0, 0))
    return {"staged_files": len(EXPECTED), "epoch": 0, "models_imported": 0}


if __name__ == "__main__":
    print(stage(Path(sys.argv[1])))
