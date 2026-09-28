"""Router settings: config, file locations, credentials."""
import json
import logging
import os
from pathlib import Path

ROOT = Path(__file__).parent
CONFIG = json.loads((ROOT / "config.json").read_text())
LOG_FILE = ROOT / os.environ.get("ROUTER_LOG", "router.log")
STATS_FILE = ROOT / os.environ.get("ROUTER_STATS", "stats.jsonl")
SESSIONS_FILE = ROOT / "sessions.json"
AUTH_FILE = Path(os.environ.get("ROUTER_AUTH_FILE", Path.home() / ".pi/agent/auth.json"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()],
)
log = logging.getLogger("router")


def gateway_key() -> str:
    return json.loads(AUTH_FILE.read_text())["vercel-ai-gateway"]["key"]


def codex_auth():
    d = json.loads(AUTH_FILE.read_text())["openai-codex"]
    tok = d["access"]["token"] if isinstance(d.get("access"), dict) else d["access"]
    return tok, d.get("accountId")
