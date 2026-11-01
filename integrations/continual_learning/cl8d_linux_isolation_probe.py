"""CL-8D fixed NO-MODEL Linux isolation probe.

This module is deliberately NOT a model runner. It accepts no user commands,
scripts, datasets or model imports. Missing network namespaces fail closed.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import resource
import stat
import signal
import subprocess
import sys

from integrations.continual_learning.cl8c_execution_envelope_audit import audit as audit_cl8c

# Fixed, auditable source. Never substitute arbitrary user input for this code.
PROBE_SOURCE = (
    "import json, os; "
    "print(json.dumps({'probe':'CL8D_NO_MODEL',"
    "'net_namespace':os.readlink('/proc/self/ns/net')},sort_keys=True))"
)
CPU_LIMIT_SECONDS = 5
ADDRESS_SPACE_LIMIT_BYTES = 256 * 1024 * 1024
WALL_LIMIT_SECONDS = 5
OUTPUT_LIMIT_BYTES = 8192


def _trusted_unshare() -> str | None:
    """Select root-owned, non-writable system binary independent of caller PATH."""
    for candidate in (Path("/usr/bin/unshare"), Path("/bin/unshare")):
        try:
            resolved = candidate.resolve(strict=True)
            if (resolved.name != "unshare"
                    or resolved.parent not in (Path("/usr/bin"), Path("/bin"))):
                continue
            info = resolved.stat()
            directory = resolved.parent.stat()
            if (not stat.S_ISREG(info.st_mode)
                    or info.st_uid != 0 or info.st_mode & 0o022
                    or directory.st_uid != 0 or directory.st_mode & 0o022
                    or not os.access(resolved, os.X_OK)):
                continue
            return str(resolved)
        except (OSError, RuntimeError):
            continue
    return None


def _child_limits() -> None:
    """Only called in the disposable pre-exec child; parent is untouched."""
    resource.setrlimit(resource.RLIMIT_CPU, (CPU_LIMIT_SECONDS, CPU_LIMIT_SECONDS))
    resource.setrlimit(resource.RLIMIT_AS, (ADDRESS_SPACE_LIMIT_BYTES, ADDRESS_SPACE_LIMIT_BYTES))
    resource.setrlimit(resource.RLIMIT_FSIZE, (OUTPUT_LIMIT_BYTES, OUTPUT_LIMIT_BYTES))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def _stop_process_tree(process: subprocess.Popen) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.communicate()


def validate_probe_response(
    returncode: int, stdout: bytes, stderr: bytes, parent_namespace: str
) -> dict:
    if not isinstance(stdout, bytes) or not isinstance(stderr, bytes):
        raise ValueError("untrusted subprocess output must be raw bytes")
    if len(stdout) > OUTPUT_LIMIT_BYTES or len(stderr) > OUTPUT_LIMIT_BYTES:
        raise ValueError("probe exceeded output budget")
    if returncode != 0:
        # Only expose a bounded, categorical diagnostic. Do not trust or
        # directly print arbitrary child stderr in logs or Markdown.
        message = stderr.decode("utf-8", errors="replace").lower()
        if "operation not permitted" in message:
            category = "OS_PERMISSION_DENIED"
        elif "permission denied" in message:
            category = "OS_PERMISSION_DENIED"
        elif "unrecognized option" in message or "invalid option" in message:
            category = "UNSHARE_OPTION_UNSUPPORTED"
        else:
            category = "CHILD_FAILURE_UNCLASSIFIED"
        raise ValueError(f"network isolation unavailable: {category}")
    try:
        response = json.loads(stdout.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid fixed probe response") from exc
    if (type(response) is not dict or set(response) != {"probe", "net_namespace"}
            or response["probe"] != "CL8D_NO_MODEL"
            or not isinstance(response["net_namespace"], str)
            or not response["net_namespace"].startswith("net:[")
            or response["net_namespace"] == parent_namespace):
        raise ValueError("network namespace separation not proved")
    return {
        "kind": "CL8D_FIXED_CHILD_PROBE_ONLY",
        "status": "ISOLATED_PROBE_OK",
        "network_namespace_separated": True,
        "cpu_limit_configured": True,
        "address_space_limit_configured": True,
        "wall_timeout_configured": True,
        "model_executed": False,
        "training_executed": False,
        "general_model_runner_available": False,
    }


def probe() -> dict:
    envelope = audit_cl8c()
    if (envelope["decision"] != "NO_GO"
            or envelope["execution_entrypoint_available"] is not False):
        raise ValueError("future experiment runner must remain disabled")
    if platform.system() != "Linux" or not Path("/proc/self/ns/net").exists():
        raise RuntimeError("Linux network namespaces unavailable; fail closed")
    network_tool = _trusted_unshare()
    if not network_tool:
        raise RuntimeError("unshare unavailable; fail closed without any subprocess")
    parent_namespace = os.readlink("/proc/self/ns/net")
    # Fixed args only; no shell, variable workload, network download or model.
    command = [
        network_tool, "--user", "--map-root-user", "--net", "--",
        sys.executable, "-I", "-S", "-c", PROBE_SOURCE,
    ]
    child = subprocess.Popen(
        command, cwd="/", stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        close_fds=True, start_new_session=True,
        preexec_fn=_child_limits, env={"PATH": "/usr/bin:/bin"},
    )
    try:
        out, err = child.communicate(timeout=WALL_LIMIT_SECONDS)
    except subprocess.TimeoutExpired as exc:
        _stop_process_tree(child)
        raise RuntimeError("fixed no-model probe exceeded wall-time limit") from exc
    if len(out) > OUTPUT_LIMIT_BYTES or len(err) > OUTPUT_LIMIT_BYTES:
        raise RuntimeError("fixed probe exceeded output byte limit")
    return validate_probe_response(child.returncode, out, err, parent_namespace)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CL-8D no-model Linux isolation probe")
    parser.add_argument("--probe", action="store_true", required=True)
    args = parser.parse_args(argv)
    try:
        result = probe()
    except (RuntimeError, OSError, ValueError, subprocess.SubprocessError) as exc:
        result = {
            "kind": "CL8D_FIXED_CHILD_PROBE_ONLY",
            "status": "NO_GO_ISOLATION_UNAVAILABLE",
            "reason": str(exc)[:300],
            "model_executed": False,
            "training_executed": False,
            "general_model_runner_available": False,
        }
    print(json.dumps(result, sort_keys=True, indent=2))
    # A missing namespace is an expected fail-closed state, never success.
    return 0 if result["status"] == "ISOLATED_PROBE_OK" else 2


if __name__ == "__main__":
    raise SystemExit(main())
