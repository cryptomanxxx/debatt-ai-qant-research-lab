"""CL-27 local Docker-save layer metadata comparison (NO MODEL).

Only reads image archives made in this temporary GitHub Actions job.
Does not import models, run containers, contact the network or create artifacts.
"""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile


def _outer(path: Path):
    with tarfile.open(path, "r:") as image:
        manifest = json.load(image.extractfile("manifest.json"))[0]
        config = json.load(image.extractfile(manifest["Config"]))
        layers = [image.extractfile(name).read() for name in manifest["Layers"]]
    return config, layers


def _entries(raw: bytes):
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:") as layer:
        return [
            {
                "path": item.name, "mtime": item.mtime, "size": item.size,
                "uid": item.uid, "gid": item.gid, "mode": item.mode,
                "pax": item.pax_headers,
                "content_sha256": hashlib.sha256(layer.extractfile(item).read()).hexdigest()
                if item.isfile() else None,
            }
            for item in layer.getmembers()
        ]


def compare(a: Path, b: Path) -> int:
    left, left_layers = _outer(a)
    right, right_layers = _outer(b)
    print("created_A", left.get("created"), "created_B", right.get("created"))
    print("layer_counts", len(left_layers), len(right_layers))
    mismatches = 0
    for i, (x, y) in enumerate(zip(left_layers, right_layers)):
        if hashlib.sha256(x).digest() == hashlib.sha256(y).digest():
            continue
        mismatches += 1
        print("DIFFERENT_LAYER", i)
        first, second = _entries(x), _entries(y)
        print("A_ENTRIES", json.dumps(first[:3], sort_keys=True))
        print("B_ENTRIES", json.dumps(second[:3], sort_keys=True))
        if mismatches >= 2:
            break
    if left != right:
        print("CONFIG_DIFFERENT", sorted(k for k in set(left) | set(right) if left.get(k) != right.get(k)))
    return mismatches


if __name__ == "__main__":
    compare(Path(sys.argv[1]), Path(sys.argv[2]))
