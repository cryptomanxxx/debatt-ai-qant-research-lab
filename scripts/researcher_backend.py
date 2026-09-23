"""Provider-neutral AI Researcher backend boundary.

This module deliberately makes no network/API calls. External model providers
can later implement propose(context) while the research pipeline keeps the same
proposal-only safety boundary.
"""
import json
from pathlib import Path

CONFIG = Path("research_queue/researcher_backend.json")
ALLOWED = {"deterministic", "groq", "codex", "codestral"}

def load_backend_config():
    cfg = json.loads(CONFIG.read_text())
    if cfg.get("backend") not in ALLOWED:
        raise ValueError("Unsupported researcher backend")
    if cfg.get("mode") not in {"disabled", "proposal-only"}:
        raise ValueError("Researcher backend must be disabled or proposal-only")
    if cfg.get("allow_paid_usage") is not False:
        raise ValueError("Paid AI usage is forbidden by researcher policy")
    return cfg

def backend_status():
    cfg = load_backend_config()
    return {
        "backend": cfg["backend"],
        "mode": cfg["mode"],
        "model": cfg.get("model"),
        "allow_paid_usage": False,
        "external_api_enabled": False
    }

if __name__ == "__main__":
    print(json.dumps(backend_status(), indent=2))
