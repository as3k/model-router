"""Session registry: one pinned tier per agent session, monotonic promotion.

A session is keyed by a hash of its system prompt: identical across every turn
of one agent run, different across agents and tasks. Pinning is the mechanism
that keeps agent sessions coherent — one brain owns the whole task chain, so
classification runs once (first turn) instead of on every message.
"""
import json
import time

from modules.settings import SESSIONS_FILE, log
from modules.classify import SUBAGENT_MARKERS

SESSION_CFG_KEY = "session"
_sessions = {}
try:
    _sessions = json.loads(SESSIONS_FILE.read_text())
except Exception:
    _sessions = {}


def save():
    try:
        SESSIONS_FILE.write_text(json.dumps(_sessions))
    except Exception as e:
        log.warning(f"sessions save failed: {e}")


def expired(sess: dict, ttl_hours: float) -> bool:
    import time
    return time.time() - sess.get("ts", 0) > ttl_hours * 3600


def detect_kind(body: dict) -> str:
    """main (interactive with the user) vs subagent/cron. Crons announce
    themselves in their system prompt."""
    sysprompt = next((str(m.get("content", "")) for m in body.get("messages", [])
                      if m.get("role") == "system"), "")
    low = sysprompt.lower()
    return "subagent" if any(mk in low for mk in SUBAGENT_MARKERS) else "main"
