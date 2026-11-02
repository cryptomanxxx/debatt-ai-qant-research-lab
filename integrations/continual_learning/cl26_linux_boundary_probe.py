"""CL-26 Linux boundary observations in a NO-MODEL candidate container.

Evidence applies only to this one Docker invocation; it does not certify
GitHub runner/kernel isolation or authorize inference or training.
"""
from __future__ import annotations

import errno
import json
import os
from pathlib import Path
import runpy
import socket

DIR = Path("/opt/research/integrations/continual_learning")
DENIED = {errno.ENETUNREACH, errno.EHOSTUNREACH, errno.EPERM, errno.EACCES}


def _status() -> dict[str, str]:
    return dict(
        line.split(":", 1) for line in
        (item.strip() for item in Path("/proc/self/status").read_text().splitlines())
        if ":" in line
    )


def _root_mount_is_readonly() -> bool:
    for line in Path("/proc/self/mountinfo").read_text().splitlines():
        fields = line.split()
        if len(fields) > 6 and fields[4] == "/":
            return "ro" in fields[5].split(",")
    return False


def verify_no_model_boundary() -> dict:
    status = _status()
    if (os.geteuid() != 65534 or os.getegid() != 65534
            or status.get("CapEff", "").strip() != "0000000000000000"
            or status.get("NoNewPrivs", "").strip() != "1"
            or not _root_mount_is_readonly()):
        raise RuntimeError("CL-26 NO_GO: uid/capabilities/privileges/read-only drift")
    group = Path("/sys/fs/cgroup")
    if not (group / "cgroup.controllers").is_file():
        raise RuntimeError("CL-26 NO_GO: expected cgroup v2 controls missing")
    if ((group / "pids.max").read_text().strip() != "32"
            or (group / "memory.max").read_text().strip() != "134217728"
            or (group / "cpu.max").read_text().strip() != "100000 100000"):
        raise RuntimeError("CL-26 NO_GO: process, memory or CPU quota drift")

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as udp:
        try:
            sent = udp.sendto(b"cl26", ("1.1.1.1", 443))
        except OSError as exc:
            if exc.errno not in DENIED:
                raise RuntimeError("CL-26 NO_GO: unexpected IPv4 UDP error") from exc
        else:
            raise RuntimeError(f"CL-26 NO_GO: IPv4 UDP unexpectedly sent {sent} bytes")

    try:
        with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as tcp6:
            tcp6.settimeout(2)
            code = tcp6.connect_ex(("2606:4700:4700::1111", 443))
            if code not in DENIED:
                raise RuntimeError(f"CL-26 NO_GO: unexpected IPv6 TCP result {code}")
    except OSError as exc:
        if exc.errno not in DENIED | {errno.EAFNOSUPPORT, errno.EPROTONOSUPPORT}:
            raise RuntimeError("CL-26 NO_GO: unexpected IPv6 socket error") from exc

    prior = runpy.run_path(str(DIR / "cl25_no_model_single_input_probe.py"))
    receipt = prior["verify_review_only"]()
    if receipt["decision"] != "NO_GO" or receipt["models_imported"] != 0:
        raise RuntimeError("CL-26 NO_GO: prior no-model probe drift")
    return {
        "decision": "NO_GO",
        "observed_uid_gid": "65534:65534",
        "effective_capabilities_zero": True,
        "no_new_privileges": True,
        "root_mount_read_only": True,
        "observed_pids_max": 32,
        "observed_memory_mib": 128,
        "observed_cpu_quota": "1",
        "ipv4_udp_egress_denied": True,
        "ipv6_tcp_egress_denied_or_unsupported": True,
        "ipv4_tcp_egress_denied": True,
        "packaged_samples": 1,
        "models_imported": 0,
        "model_inference_authorized": False,
        "training_authorized": False,
        "physical_energy": "NOT_MEASURED",
        "host_isolation_independently_approved": False,
    }


if __name__ == "__main__":
    print(json.dumps(verify_no_model_boundary(), sort_keys=True))
