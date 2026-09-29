"""One authorized Dev-4 GPT-OSS selection; no evaluation, retries or training."""
import hashlib
import json
import os
from pathlib import Path
import urllib.error
import urllib.request

from scripts.dev4_policy_adapters import choose_grid, choose_random, choose_bayesian, choose_gpt_external
from scripts.dev4_selection_dry_run import HISTORY, PROMPT, PROTOCOL, verify_inputs


def select_once(output_dir, opener=urllib.request.urlopen):
    protocol = json.loads(PROTOCOL.read_text())
    history_bytes, prompt_bytes = HISTORY.read_bytes(), PROMPT.read_bytes()
    history = verify_inputs(protocol, history_bytes, prompt_bytes)
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        raise RuntimeError("GROQ_API_KEY is not configured; no API call made")
    # The frozen snapshot is the ONLY shared history; no future holdout or
    # other strategies' proposal or evaluation outcomes are included.
    context = {"shared_initial_history_snapshot": history,
               "own_proposal_status_ledger": [],
               "own_completed_paired_outcomes": [],
               "round": 1,
               "allowed_local_widths": protocol["search_space"]["candidate_local_widths"]}
    payload = {"model": protocol["strategy_specifications"]["gpt-oss-120b"]["model_identifier"],
               "temperature": 0, "max_completion_tokens": 512,
               "messages": [{"role": "system", "content": prompt_bytes.decode("utf-8")},
                            {"role": "user", "content": json.dumps(context, sort_keys=True)}]}
    request = urllib.request.Request(
        "https://api.groq.com/openai/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"), method="POST",
        headers={"Authorization": "Bearer " + key, "Content-Type": "application/json",
                 "Accept": "application/json", "User-Agent": "debatt-ai-qant-research-lab/dev4"})
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    audit = {"schema_version": 1, "mode": "first_live_groq_selection_no_compute",
             "history_sha256": "sha256:" + hashlib.sha256(history_bytes).hexdigest(),
             "prompt_sha256": "sha256:" + hashlib.sha256(prompt_bytes).hexdigest(),
             "request_parameters": {"model": payload["model"], "temperature": 0,
                                    "max_completion_tokens": 512},
             "request_payload_sha256": "sha256:" + hashlib.sha256(
                 json.dumps(payload, sort_keys=True).encode()).hexdigest(),
             "training_runs": 0, "holdout_access": False, "groq_calls": 1}
    try:
        with opener(request, timeout=90) as response:
            body = json.load(response)
        choice = body["choices"][0]
        raw = choice["message"].get("content") or ""
        width, error = choose_gpt_external(protocol, raw, [])
        audit.update({"response_id": body.get("id"), "response_model": body.get("model"),
                      "created": body.get("created"), "system_fingerprint": body.get("system_fingerprint"),
                      "usage": body.get("usage"), "finish_reason": choice.get("finish_reason"),
                      "raw_model_response": raw, "gpt_local_width": width,
                      "gpt_proposal_error": error,
                      "other_first_round_selections": {
                          "grid-search": choose_grid(protocol, 1),
                          "random-search": choose_random(protocol, 1),
                          "bayesian-optimization": choose_bayesian(protocol, [], [])}})
    except (urllib.error.HTTPError, urllib.error.URLError) as exc:
        audit["api_error"] = {"type": type(exc).__name__, "http_status": getattr(exc, "code", None)}
        raise
    finally:
        (output_dir / "dev4_first_live_selection.json").write_text(
            json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"gpt_local_width": audit.get("gpt_local_width"),
                      "gpt_proposal_error": audit.get("gpt_proposal_error"),
                      "training_runs": 0, "artifact": "dev4_first_live_selection.json"}, sort_keys=True))
    return audit


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    select_once(args.output_dir)
