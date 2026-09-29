"""Build a proposal lineage manifest from archived, explicit-ID proposal snapshots."""
import argparse
import json
from pathlib import Path
from scripts.register_research_proposal import ARCHIVE


def build_manifest(archive=ARCHIVE):
    entries = []
    seen = set()
    for path in sorted(Path(archive).glob("research-*.json")):
        draft = json.loads(path.read_text(encoding="utf-8"))
        pid = draft.get("proposal_id")
        if not isinstance(pid, str) or path.stem != pid or pid in seen:
            raise ValueError("invalid or duplicate archived proposal ID: " + str(path))
        if draft.get("status") != "draft_requires_human_review":
            raise ValueError("invalid archived draft status: " + str(path))
        seen.add(pid)
        entries.append({"proposal_id": pid, "proposal_file": str(path),
                        "strategy": draft.get("strategy", "unclassified")})
    return entries


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive", type=Path, default=ARCHIVE)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    rendered = json.dumps(build_manifest(args.archive), indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
