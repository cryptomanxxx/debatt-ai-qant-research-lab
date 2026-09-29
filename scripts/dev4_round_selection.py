"""Dev-4 selection-only runner. Round 2 is enabled; future rounds fail closed until audited histories exist."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import urllib.error
import urllib.request

from scripts.dev4_selection_dry_run import HISTORY, PROMPT, PROTOCOL, verify_inputs
from scripts.dev4_policy_adapters import choose_grid, choose_random, choose_bayesian, choose_gpt_external

ROOT = Path(__file__).resolve().parents[1]
ROUND2 = ROOT / "research_queue/benchmarks/dev4_round2_isolated_contexts.json"
SOURCE_SHA = "43eca34825b363497440717dea1af56fc3e00b9c01cac2f39f76d298eec10172"


def prepare(round_number=2):
    if type(round_number) is not int or round_number != 2:
        raise ValueError("Only round 2 is enabled; rounds 3-5 require independently verified own-history artifacts")
    protocol = json.loads(PROTOCOL.read_text())
    history_bytes, prompt_bytes = HISTORY.read_bytes(), PROMPT.read_bytes()
    history = verify_inputs(protocol, history_bytes, prompt_bytes)
    raw = ROUND2.read_bytes()
    contexts = json.loads(raw)
    if contexts["mode"] != "round_two_feedback_only_no_selection_no_compute":
        raise ValueError("wrong context mode")
    if contexts["shared_initial_history_sha256"] != protocol["information_policy"]["shared_initial_history_snapshot"]:
        raise ValueError("frozen initial history mismatch")
    if contexts["source"]["workflow_run_id"] != 36559946601 or contexts["source"]["artifact_id"] != 11029104019 or contexts["source"]["artifact_sha256"] != SOURCE_SHA:
        raise ValueError("round 1 source mismatch")
    if set(contexts["contexts"]) != set(protocol["strategies"]):
        raise ValueError("strategy context set mismatch")
    from scripts.dev4_round1_isolated_feedback import build_contexts, FEEDBACK
    feedback = json.loads(FEEDBACK.read_text())
    if contexts != build_contexts(protocol, feedback):
        raise ValueError("round 2 context differs from source-validated feedback")
    own = contexts["contexts"]["gpt-oss-120b"]
    ledger = own["proposal_status_ledger"]
    outcomes = own["completed_paired_outcomes"]
    if ledger != [{"round": 1, "local_width": 9, "valid": True, "status": "evaluated"}]:
        raise ValueError("GPT own ledger mismatch")
    if outcomes != [{"round": 1, "paired_outcome": {"candidate_correct": 445, "control_correct": 452,
                                                       "candidate_parameters": 4812, "control_parameters": 8550}}]:
        raise ValueError("GPT own outcome mismatch")
    context = {"shared_initial_history_snapshot": history,
               "own_proposal_status_ledger": ledger,
               "own_completed_paired_outcomes": outcomes,
               "round": round_number,
               "allowed_local_widths": protocol["search_space"]["candidate_local_widths"]}
    # Never serialize the complete four-strategy contexts into the GPT request.
    payload = {"model": protocol["strategy_specifications"]["gpt-oss-120b"]["model_identifier"],
               "temperature": 0, "max_completion_tokens": 512,
               "messages": [{"role": "system", "content": prompt_bytes.decode("utf-8")},
                            {"role": "user", "content": json.dumps(context, sort_keys=True)}]}
    return protocol, contexts, payload, history_bytes, prompt_bytes


def select(round_number, output_dir, execute=False, opener=urllib.request.urlopen):
    protocol, contexts, payload, history_bytes, prompt_bytes = prepare(round_number)
    own = contexts["contexts"]
    selections = {"random-search": choose_random(protocol, round_number),
                  "grid-search": choose_grid(protocol, round_number),
                  "bayesian-optimization": choose_bayesian(
                      protocol, own["bayesian-optimization"]["proposal_status_ledger"],
                      own["bayesian-optimization"]["completed_paired_outcomes"])}
    import numpy
    import sklearn
    audit = {"adapter_versions": {"numpy": numpy.__version__, "scikit_learn": sklearn.__version__},
             "schema_version": 1, "round": round_number, "mode": "selection_only_no_qant_compute",
             "source_round1_run_id": 36559946601, "source_round1_artifact_sha256": SOURCE_SHA,
             "history_sha256": "sha256:" + hashlib.sha256(history_bytes).hexdigest(),
             "prompt_sha256": "sha256:" + hashlib.sha256(prompt_bytes).hexdigest(),
             "round2_context_sha256": "sha256:" + hashlib.sha256(ROUND2.read_bytes()).hexdigest(),
             "request_payload_sha256": "sha256:" + hashlib.sha256(
                 json.dumps(payload, sort_keys=True).encode()).hexdigest(),
             "request_parameters": {"model": payload["model"], "temperature": 0, "max_completion_tokens": 512},
             "other_selections": selections, "training_runs": 0, "holdout_access": False, "groq_calls": 0}
    if execute:
        key = os.environ.get("GROQ_API_KEY")
        if not key:
            raise RuntimeError("GROQ_API_KEY missing; no API call made")
        # Record one attempted request before network activity. Never retry.
        audit["groq_calls"] = 1
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        try:
            request = urllib.request.Request("https://api.groq.com/openai/v1/chat/completions",
                data=json.dumps(payload).encode(), method="POST",
                headers={"Authorization": "Bearer " + key, "Content-Type": "application/json",
                         "Accept": "application/json", "User-Agent": "debatt-ai-qant-research-lab/dev4"})
            with opener(request, timeout=90) as response:
                response_bytes = response.read()
            audit["raw_http_response_sha256"] = "sha256:" + hashlib.sha256(response_bytes).hexdigest()
            audit["raw_http_response_utf8"] = response_bytes.decode("utf-8", errors="replace")
            try:
                body = json.loads(response_bytes)
                choice = body["choices"][0]
                if not isinstance(choice, dict) or not isinstance(choice.get("message"), dict):
                    raise ValueError("missing choice message")
                raw = choice["message"].get("content")
                if not isinstance(raw, str):
                    raise ValueError("missing string choice content")
            except (ValueError, TypeError, KeyError, IndexError) as exc:
                audit["response_parse_error"] = type(exc).__name__ + ": " + str(exc)
                audit["gpt_local_width"] = None
                audit["gpt_proposal_error"] = "malformed_api_response"
            else:
                width, error = choose_gpt_external(
                    protocol, raw, own["gpt-oss-120b"]["proposal_status_ledger"])
                audit.update({"response_id": body.get("id"), "response_model": body.get("model"),
                              "created": body.get("created"), "system_fingerprint": body.get("system_fingerprint"),
                              "usage": body.get("usage"), "finish_reason": choice.get("finish_reason"),
                              "raw_model_response": raw, "gpt_local_width": width, "gpt_proposal_error": error})
        except (urllib.error.HTTPError, urllib.error.URLError) as exc:
            audit["api_error"] = {"type": type(exc).__name__, "http_status": getattr(exc, "code", None)}
            raise
        finally:
            (output_dir / "dev4_round2_selection.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--execute", action="store_true", help="Make one authorized Groq request; never Q.ANT compute")
    args = parser.parse_args()
    result = select(args.round, args.output_dir, execute=args.execute)
    if not args.execute:
        print("Validated isolated context and deterministic baselines; no Groq request or training.")
    print(json.dumps({"round": args.round, "groq_calls": result["groq_calls"],
                      "training_runs": result["training_runs"], "other_selections": result["other_selections"]},
                     sort_keys=True))


if __name__ == "__main__":
    main()
