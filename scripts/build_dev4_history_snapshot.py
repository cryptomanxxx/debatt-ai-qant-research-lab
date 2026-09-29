"""Build a deterministic, read-only Dev-4 history snapshot. Never authorizes compute.

Run against a frozen repository checkout, then commit the output and its SHA-256.
Exclude evaluation-seed overlap to avoid direct benchmark leakage.
"""
import argparse
import hashlib
import json
from pathlib import Path
from scripts.measure_researcher_selection import summarize


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")


def build_snapshot(files, protocol, source_root=None):
    files = [Path(path) for path in files]
    # A stable results-root-relative identifier avoids checkout-specific prefixes.
    source_root = Path(source_root).resolve() if source_root is not None else (files[0].resolve().parent if files else Path.cwd().resolve())
    eval_seeds = set(protocol["evaluation"]["seeds"])
    included, excluded = [], []
    for path in sorted(files, key=lambda p: p.resolve().relative_to(source_root).as_posix()):
        source = path.resolve().relative_to(source_root).as_posix()
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        try:
            record = json.loads(raw)
            observations = summarize(record)
        except (ValueError, TypeError, KeyError):
            observations = None
        reason = None
        if not observations:
            reason = "invalid_or_nonpaired_schema"
        elif set(observations[0]["seeds"]) & eval_seeds:
            reason = "evaluation_seed_overlap"
        elif (observations[0]["protocol"]["dataset_name"] != protocol["dataset"]["name"]
              or observations[0]["protocol"]["train_shape"] != protocol["dataset"]["train_shape"]
              or observations[0]["protocol"]["test_shape"] != protocol["dataset"]["test_shape"]):
            reason = "different_dataset_or_shape"
        if reason:
            excluded.append({"source": source, "sha256": digest, "reason": reason})
            continue
        included.append({"source": source, "sha256": digest,
                         "observations": observations})
    return {"schema_version": 1, "purpose": "common pre-benchmark history, not holdout evidence",
            "evaluation_seeds_excluded": sorted(eval_seeds),
            "included": included, "excluded": excluded}


def verify_snapshot(snapshot, files, protocol, source_root=None):
    rebuilt = build_snapshot(files, protocol, source_root=source_root)
    if canonical(snapshot) != canonical(rebuilt):
        raise ValueError("history snapshot differs from source files or protocol")
    return hashlib.sha256(canonical(snapshot)).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("pnn-v1/results"))
    parser.add_argument("--protocol", type=Path, default=Path("research_queue/benchmarks/dev4_selection_protocol.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    # Never treat the generated snapshot itself as an input, including --verify.
    files = sorted(p for p in args.results.glob("*.json")
                   if p.resolve() != args.output.resolve())
    if not files:
        parser.error("no source results; refusing empty snapshot")
    if args.verify:
        snapshot = json.loads(args.output.read_text())
        digest = verify_snapshot(snapshot, files, protocol, source_root=args.results)
    else:
        snapshot = build_snapshot(files, protocol, source_root=args.results)
        if not snapshot["included"]:
            parser.error("no eligible non-leaking paired results; cannot pin a usable snapshot")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(canonical(snapshot))
        digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    print("sha256:" + digest)
    print("included:", len(snapshot["included"]), "excluded:", len(snapshot["excluded"]))
    print("NOTE: snapshot pinning does not authorize benchmark execution.")


if __name__ == "__main__":
    main()
