"""Classification layer: Laya + Jev deciders, routing policy, and the gate.

Everything that turns "a request" into "a tier" lives here. Sessions are
classified once (first turn) and pinned by the session registry; per-turn
guards (overflow, stakes, tags) are applied by the request handlers in
router.py because they depend on the real payload.
"""
import asyncio
import hashlib
import json
import os
import re
import warnings

from laya import Router

from questions import LAYA_QUESTIONS, GATE_QUESTIONS, MICRO_QUESTIONS
from settings import CONFIG, log, gateway_key, codex_auth

warnings.filterwarnings("ignore")

RANKS = ["k2", "deepseek", "luna", "terra"]
FORCE_TIERS = RANKS + ["glm"]

TAG_RE = re.compile(r"@(k2|deepseek|glm|luna|terra)\b", re.IGNORECASE)
BRAIN_RE = re.compile(r"brain\s*[:=]\s*(k2|deepseek|glm|luna|terra)\b", re.IGNORECASE)

router_laya = Router(preload=["english"], device=os.environ.get("LAYA_DEVICE", "cuda"))


async def laya_decide(state: str) -> dict:
    """Laya decision on the tier/complexity questions. laya predict is
    synchronous CPU/GPU work, so run in a thread so parallel subagent requests
    don't serialize behind each other's classification."""
    return await asyncio.to_thread(_laya_decide_sync, state)


def _laya_decide_sync(state: str) -> dict:
    res = router_laya.predict(state, LAYA_QUESTIONS)
    a = res["answers"]
    t = a["tier"]
    tier_conf = t.get("answer_confidence") or t.get("probabilities", {}).get(t["choice"], 0)
    return {
        "tier": t["choice"],
        "tier_conf": tier_conf,
        "tier_probs": t.get("probabilities"),
        "complexity": a["complexity"]["score"],
        "cx_conf": a["complexity"]["confidence"],
        "via": "laya",
    }


async def jev_decide(state: str) -> dict:
    r = await client.post(
        CONFIG["jev"]["url"],
        headers={"Authorization": f"Bearer {gateway_key()}"},
        json={
            "model": CONFIG["jev"]["model"],
            "state": state,
            "questions": {
                "tier": {
                    "type": "choice",
                    "instructions": "What level of model capability does answering this request well actually require?",
                    "criteria": {k: v for k, v in LAYA_QUESTIONS["tier"]["criteria"].items()},
                }
            },
        },
        timeout=30,
    )
    a = r.json()["answers"]["tier"]
    return {
        "tier": a["choice"],
        "tier_conf": a["probabilities"][a["choice"]],
        "complexity": None,
        "cx_conf": None,
        "via": "jev",
    }


client = None  # shared httpx AsyncClient, set by router.py

# ---------------------------------------------------------------- sessions
SESSION_CFG = CONFIG.get("session", {"ttl_hours": 4, "reeval_turns": 25})
SUBAGENT_MARKERS = ["cron job", "scheduled", "subagent", "worker", "delegated",
                    "you are running as"]


def session_key(body: dict) -> str | None:
    """Stable per-session key: hash of the system prompt (identical across all
    turns of one agent session; differs across agents/tasks)."""
    for m in body.get("messages", []):
        if m.get("role") == "system":
            c = m.get("content")
            if isinstance(c, list):
                c = " ".join(p.get("text", "") for p in c if isinstance(p, dict))
            if c:
                return hashlib.sha256(str(c).encode()).hexdigest()[:12]
    return None


def last_user_msg(body: dict) -> str:
    for m in reversed(body.get("messages", [])):
        if m.get("role") == "user":
            c = m.get("content")
            return c if isinstance(c, str) else " ".join(
                p.get("text", "") for p in c if isinstance(p, dict))
    return ""


async def micro_delegate(body: dict, home_tier: str) -> dict | None:
    """Cheap single-question check on short turns inside main sessions:
    should this one turn be delegated down to k2? Only fires on plausible
    shapes (short user text, no tool results, no task-continuation words)."""
    lmsg = last_user_msg(body)
    if not lmsg or len(lmsg) > SESSION_CFG.get("micro_turn_chars", 300):
        return None
    if any(m.get("role") == "tool" for m in body.get("messages", [])):
        return None  # tool-result turns are never light
    if any(mk in lmsg.lower() for mk in ("continue", "step ", "analyze", "design", "review", "implement")):
        return None  # task-continuation words mean stay home
    try:
        def _ask():
            a = router_laya.predict(lmsg, MICRO_QUESTIONS)["answers"]["delegate"]
            conf = a.get("answer_confidence") or a.get("probabilities", {}).get(a["choice"], 0)
            return a["choice"], conf
        choice, conf = await asyncio.to_thread(_ask)
        if choice == "k2" and conf >= SESSION_CFG.get("micro_floor", 0.6):
            return {"tier": "k2", "via": "micro-delegate", "conf": conf}
    except Exception as e:
        log.warning(f"micro-delegate check failed: {e}")
    return None


# ---------------------------------------------------------------- decide
async def decide(state: str, force: str | None = None, est_tokens: int = 0) -> dict:
    if force and force in FORCE_TIERS:
        return {"tier": force, "tier_conf": 1.0, "complexity": None, "cx_conf": None, "via": "forced"}
    try:
        d = await laya_decide(state)
    except Exception as e:  # laya down -> jev -> default
        log.warning(f"laya failed: {e}")
        try:
            d = await jev_decide(state)
        except Exception as e2:
            log.warning(f"jev failed: {e2}")
            return {"tier": CONFIG["default_tier"], "tier_conf": 0.0, "complexity": None,
                    "cx_conf": None, "via": "default"}
    floor = CONFIG["confidence_floor"]
    if d["tier_conf"] < floor and d["via"] == "laya":
        try:
            j = await jev_decide(state)
            d["jev"] = {k: j[k] for k in ("tier", "tier_conf")}
            if j["tier_conf"] >= d["tier_conf"]:
                # Jev is a tiebreaker, not a takeover: its distributions are
                # razor-sharp (prob 1.00 even on noise like "testing testing"),
                # so clamp its vote to at most one tier above laya's choice.
                laya_rank = RANKS.index(d["tier"])
                jev_rank = min(RANKS.index(j["tier"]), laya_rank + 1)
                d["tier"] = RANKS[jev_rank]
                d["tier_conf"] = j["tier_conf"]
                d["via"] = "jev"
                d["laya_tier"] = RANKS[laya_rank]
        except Exception as e:
            log.warning(f"jev fallback failed: {e}")
    # complexity can only escalate, never de-escalate. Thresholds are relative
    # to laya's chosen tier (its score primitive is noisy on trivial input, so
    # a confident tier vote wins over a middling complexity score).
    cx = d.get("complexity")
    if cx is not None:
        thr = CONFIG["cx_escalation"]  # per current rank: [k2->ds, ds->luna, luna->terra]
        rank = RANKS.index(d["tier"])
        while rank < 3 and cx >= thr[rank]:
            rank += 1
        if rank > RANKS.index(d["tier"]):
            d["tier"] = RANKS[rank]
            d["escalated_by"] = "complexity"
    # NOTE: length-based k2 escalation lives in the request handlers (chat/
    # routing_probe) because it depends on the real payload size incl. tools.
    return d


# ---------------------------------------------------------------- gate
def _gate_laya_sync(state: str):
    """Sync wrapper: laya predict is CPU/GPU work, so run in a thread."""
    res = router_laya.predict(state, GATE_QUESTIONS)
    a = res["answers"]["notify"]
    conf = a.get("answer_confidence") or a.get("probabilities", {}).get(a["choice"], 0)
    return {"label": a["choice"], "conf": conf, "via": "laya"}


async def gate_laya(state: str):
    return await asyncio.to_thread(_gate_laya_sync, state)


async def gate_jev(state: str):
    r = await client.post(
        CONFIG["jev"]["url"],
        headers={"Authorization": f"Bearer {gateway_key()}"},
        json={
            "model": CONFIG["jev"]["model"],
            "state": state,
            "questions": {"notify": GATE_QUESTIONS["notify"]},
        },
        timeout=30,
    )
    a = r.json()["answers"]["notify"]
    return {"label": a["choice"], "conf": a["probabilities"][a["choice"]], "via": "jev"}


# Deterministic event markers: these words unambiguously mean "something
# happened" regardless of surrounding context ("back up, healthy now" after
# "Exited" must NOT classify as silent). Checked before the classifier so the
# cheap rule wins and the model only arbitrates genuinely ambiguous text.
GATE_EVENT_MARKERS = [
    "exited", "dead", "unhealthy", "restart", "oom", "critical", "failed",
    "failure", "error", "down", "stopped", "investigat", "degraded",
    "unreachable", "threshold", "warning", "alert", "incident", "outage",
]

# Negation preceding a marker means the event did NOT happen: "no alerts",
# "not down", "no restart needed". Skip that hit.
_NEG_RE = re.compile(r"\b(?:no|not|never|none|without)\b(?:\s+\w+){0,3}\s*$")


def _marker_hits(state_lower: str) -> list:
    hits = []
    for m in GATE_EVENT_MARKERS:
        start = 0
        while True:
            i = state_lower.find(m, start)
            if i == -1:
                break
            before = state_lower[max(0, i - 20):i]
            if not _NEG_RE.search(before):
                hits.append(m)
            start = i + 1
    return hits


async def gate_decide(state: str) -> dict:
    """Classify an alert as notify / silent. Fail-open: on classifier failure
    the alert MUST still go through. A dead gate must never swallow an outage.
    Silence requires confidence: a low-confidence 'silent' verdict notifies
    instead (redundant ping is fine; a swallowed outage is not).
    Deterministic event markers short-circuit to notify before the model."""
    low = state.lower()
    hits = _marker_hits(low)
    if hits:
        return {"label": "notify", "conf": 1.0, "via": "rule", "markers": hits}
    try:
        d = await gate_laya(state)
    except Exception as e:  # laya down -> jev -> fail-open
        log.warning(f"gate laya failed: {e}")
        try:
            d = await gate_jev(state)
        except Exception as e2:
            log.warning(f"gate jev failed: {e2}")
            return {"label": "notify", "conf": 0.0, "via": "default"}  # fail-open default
    floor = CONFIG["confidence_floor"]
    if d["conf"] < floor:
        try:
            j = await gate_jev(state)
            if j["conf"] > d["conf"]:
                d = j
        except Exception as e:
            log.warning(f"gate jev fallback failed: {e}")
    # Silence must be confident. Notify on anything that isn't a confident silent.
    if d["label"] == "silent" and d["conf"] >= floor:
        return d
    return {"label": "notify", "conf": d["conf"], "via": d["via"]}


# ---------------------------------------------------------------- per-turn policy
def declared_brain(body: dict) -> str | None:
    """Explicit brain declaration in the system prompt: a line like
    'brain: terra' or 'brain=k2'. Wins over everything except client-pinned
    models and @tags (those are per-turn, this is per-session)."""
    for m in body.get("messages", []):
        if m.get("role") != "system":
            continue
        c = m.get("content")
        if isinstance(c, list):
            c = " ".join(p.get("text", "") for p in c if isinstance(p, dict))
        if not c:
            continue
        m2 = BRAIN_RE.search(str(c))
        if m2:
            return m2.group(1).lower()
    return None


def rule_brain(body: dict) -> str | None:
    """Name-based brain rules: config session_brain_rules = [{match, tier}],
    first substring hit on the system prompt wins."""
    sysprompt = " ".join(str(m.get("content", "")) for m in body.get("messages", [])
                          if m.get("role") == "system").lower()
    if not sysprompt:
        return None
    for rule in CONFIG.get("session_brain_rules", []):
        if rule.get("match", "").lower() in sysprompt and rule.get("tier") in RANKS + ["glm"]:
            return rule["tier"]
    return None


def detect_tag(messages: list) -> str | None:
    """Inline tier directive: @k2/@deepseek/@luna/@terra anywhere in the last
    user message forces that tier for this request only (midmessage supported).
    Last tag in the message wins."""
    for m in reversed(messages):
        if m.get("role") != "user":
            continue
        c = m.get("content")
        text = c if isinstance(c, str) else " ".join(
            p.get("text", "") for p in c if isinstance(p, dict))
        tags = TAG_RE.findall(text or "")
        if tags:
            return tags[-1].lower()
        break  # only the most recent user message is scanned
    return None


# ---------------------------------------------------------------- format helpers
def state_from_messages(messages: list, limit: int = 1600) -> str:
    """Build the classifier state from the conversation, excluding system prompts
    (agent tool instructions would otherwise dominate the classification)."""
    convo = [m for m in messages if m.get("role") != "system"]
    parts = []
    for m in convo[-3:]:
        c = m.get("content")
        if isinstance(c, list):  # multimodal -> text parts only
            c = " ".join(p.get("text", "") for p in c if isinstance(p, dict))
        parts.append(f"{m.get('role','user')}: {str(c)}")
    text = "\n".join(parts)
    return text[-limit:]