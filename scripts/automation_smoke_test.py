"""Minimal Q.ANT automation smoke test.

Purpose: verify GitHub Actions -> Q.ANT CPU backend -> local_results -> GitHub.
This is infrastructure validation, not a scientific benchmark.
"""
import json, platform
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from ml_dtypes import bfloat16
import qant_native_computing_toolkit.ai as q_ai

x=np.array([1.0,-2.0,3.0],dtype=bfloat16)
y=np.asarray(q_ai.relu_fprop(x))
expected=np.array([1.0,0.0,3.0])
passed=bool(np.allclose(y.astype(float),expected))

result={
 "schema_version":1,
 "test_id":"automation_smoke_test",
 "timestamp_utc":datetime.now(timezone.utc).isoformat(),
 "backend":"qant-cpu",
 "python":platform.python_version(),
 "operation":"relu_fprop",
 "input":[1.0,-2.0,3.0],
 "output":[float(v) for v in y],
 "expected":[1.0,0.0,3.0],
 "passed":passed,
 "notes":"Infrastructure smoke test only; not a performance benchmark."
}
Path("local_results").mkdir(exist_ok=True)
Path("local_results/automation_smoke_test.json").write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps(result,indent=2))
if not passed: raise SystemExit("Q.ANT smoke test failed")
