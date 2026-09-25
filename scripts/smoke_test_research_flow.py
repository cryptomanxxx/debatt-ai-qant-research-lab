"""Cheap static smoke tests for research-flow safety."""
import ast
from pathlib import Path

for root in (Path("pnn-v1/experiments"),Path("scripts")):
 for p in root.glob("**/*.py"):
  ast.parse(p.read_text(),filename=str(p))

# Regression check for the Exp021 bug: candidate local widths must really affect
# both local blocks and readout dimensions.
p=Path("pnn-v1/experiments/exp021/run.py")
if p.exists():
 s=p.read_text()
 for token in ("def __init__(self,local_width,nheads)","FourierBlock(16,local_width)","FourierBlock(6*local_width,2)","for cid,(local_width,nheads) in CANDIDATES.items()","train(seed,Xtr,ytr,local_width,nheads)"):
  assert token in s, f"Exp021 wiring regression: missing {token}"
print("Static research smoke tests passed")
