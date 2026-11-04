"""CL-74: CPU baselines and a fidelity check of the pilot kit (Tier 0).

Run from the repository root after prepare.py: python -m experiments.cl74_pilot_kit.run
Each thread count runs in its own process, because BLAS reads its thread count at import.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

from experiments.cl74_pilot_kit import kit

HERE = Path(__file__).resolve().parent


def bench(threads):
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "bench.json"
        subprocess.run([sys.executable, "-m", "experiments.cl74_pilot_kit.kit", "bench", "--backend", "float32",
                        "--threads", str(threads), "--out", str(out)], check=True, stdout=subprocess.DEVNULL)
        return json.loads(out.read_text())


def main():
    res = {"experiment": "CL-74", "cpu": subprocess.run(["sh", "-c", "lscpu | grep 'Model name'"], capture_output=True,
                                                        text=True).stdout.split(":")[-1].strip(),
           "gpu": "NOT_MEASURED (no GPU in this environment)", "energy": "NOT_MEASURED (no power counter or instrument)",
           "fidelity": {name: kit.check(kit.BACKENDS[name]()) for name in ("float32", "qant_sim")},
           "bench": [bench(1), bench(4)]}
    (HERE / "results.json").write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps(res["fidelity"], indent=2))
    for b in res["bench"]:
        for r in b["rows"]:
            print(b["threads"], r["workload"], r["batch"], round(r["latency_ms_median"], 3), round(r["gmac_per_s"], 2),
                  round(r["matrix_time_share"], 2), round(r["weight_load_ms"], 3))


if __name__ == "__main__":
    main()
