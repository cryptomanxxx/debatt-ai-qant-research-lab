"""Freeze a retrieval corpus from the repository's own Markdown at a fixed commit (Tier 0: data already in the repo).

Run from the repository root: python -m experiments.cl73_search_embeddings.build_corpus
"""
from __future__ import annotations

import gzip
import json
import re
import subprocess
from pathlib import Path

COMMIT = "27970ec53a4cab7fff077bd04fb99a9122841469"  # main after CL-72 (#345)
OUT = Path(__file__).resolve().parent / "corpus.json.gz"


def words(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def main():
    files = subprocess.check_output(["git", "ls-tree", "-r", "--name-only", COMMIT], text=True).split()
    paragraphs, seen = [], []
    for f in sorted(x for x in files if x.endswith(".md")):
        text = subprocess.check_output(["git", "show", f"{COMMIT}:{f}"], text=True, errors="ignore")
        for p in re.split(r"\n\s*\n", text):
            w = words(p)
            if not 30 <= len(w) <= 400:
                continue
            s = set(w)
            # drop near-duplicates (the repo repeats summaries across files)
            if any(len(s & t) / len(s | t) > 0.6 for t in seen):
                continue
            seen.append(s)
            paragraphs.append({"file": f, "words": w[:120]})
    with gzip.open(OUT, "wt", encoding="utf-8") as fh:
        json.dump({"commit": COMMIT, "paragraphs": paragraphs}, fh)
    print(len(paragraphs), "paragraphs")


if __name__ == "__main__":
    main()
