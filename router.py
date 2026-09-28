#!/usr/bin/env python3
"""Model Router: a decision layer that sits between my agent fleet and its models.

I run a bunch of AI agents (cron jobs, interactive assistants, subagents) on a
Tailscale-networked homelab. Before this router, every request went to whichever
model the agent was configured with, which meant two problems: I paid frontier
prices for turns like "what's 2+2", and when I tried classifying each message
separately, agent sessions bounced between models mid-task and lost the thread
of their own work.

The fix has three parts:

1. Laya, a 421M "System 1" decision model (local, CPU, ~35ms per forward pass).
   It never generates text. It answers typed questions about a request with
   calibrated probabilities: what capability tier does this need, how complex is
   it, and what are the stakes if it's wrong.
2. Jev, a commercial decision model through Vercel AI Gateway, as a tiebreaker
   when Laya is unsure. It also powers /v1/gate, a notify/silent triage endpoint
   for watchdog alerts.
3. A tiered fleet with escalation-only semantics:
   k2 (local 4B llama.cpp server) -> deepseek-v4-flash -> gpt-5.6-luna -> gpt-5.6-terra

Design decisions worth calling out (each one earned in production, details inline):

- Session pinning: classify once per session, then 0ms routing for every turn
  after. Agents need one coherent brain per task, not the best model per message.
- Micro-delegation: short self-contained interjections get handled one-off by
  the local model without disturbing the session's pinned brain.
- Brain declarations: crons declare their tier with a "brain: terra" line in
  their system prompt. Laya judges complexity; only the human knows stakes.
- Jev is a tiebreaker, not a takeover. Its vote is clamped to one tier above
  Laya's choice, because its distributions are so confident it once returned
  p=1.00 on the string "testing testing".
- Fail-open watchdog gate. A redundant ping is noise; a swallowed outage is an
  incident. The gate can over-notify, never under-notify.
- Overflow rescue and fallback chains. Requests that can't fit the local model
  get rescued to the next tier, and dead providers degrade instead of failing.

Every decision is logged twice: a human-readable line (router.log) and a JSON
record (stats.jsonl) with the full Laya/Jev probability distributions, the
classified text excerpt, and latency. Aggregates, recent decisions, and cost
estimates are served over HTTP (/v1/stats, /v1/recent).

Runs as a systemd user service on my homelab box, reachable only over
Tailscale. The tailnet is the trust boundary; the router has no auth of its
own. Personal paths are parameterized via ROUTER_* env vars below.
"""
import asyncio
import json
import logging
import os
import re
import time
from pathlib import Path

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

ROOT = Path(__file__).parent
CONFIG = json.loads((ROOT / "config.json").read_text())
LOG_FILE = ROOT / os.environ.get("ROUTER_LOG", "router.log")
STATS_FILE = ROOT / os.environ.get("ROUTER_STATS", "stats.jsonl")
AUTH_FILE = Path(os.environ.get("ROUTER_AUTH_FILE", Path.home() / ".pi/agent/auth.json"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()],
)
log = logging.getLogger("router")

# ---------------------------------------------------------------- laya
from laya import Router  # noqa: E402
import warnings  # noqa: E402

warnings.filterwarnings("ignore")

LAYA_QUESTIONS = {
    "tier": {
        "type": "choice",
        "instructions": "What level of model capability does answering this request well actually require?",
        "criteria": {
            "k2": "short, simple, or mechanical: quick facts, rewrites, small edits, chit-chat, formatting",
            "deepseek": "moderate: multi-step reasoning, code explanation, drafting, summarizing documents",
            "luna": "hard: complex code, architecture, nuanced analysis, needs a frontier model but speed matters",
            "terra": "hardest: deep multi-file reasoning, subtle debugging, high-stakes or very long-chain tasks",
        },
    },
    "complexity": {
        "type": "score",
        "instructions": "How cognitively complex is this request?",
        "criteria": ["trivial", "simple", "moderate", "complex", "very complex"],
    },
}

GATE_QUESTIONS = {
    "notify": {
        "type": "choice",
        "instructions": "This message comes from a production watchdog or monitoring system. Decide whether it should ping the human. The discriminator is EVENTS, not severity: notify whenever anything happened — an action was taken, a container exited or restarted, something was investigated, a problem was found, a threshold was crossed, anything differs from the normal steady state. Only mark silent for pure status messages where nothing happened at all.",
        "criteria": {
            "notify": "an event occurred: container exited/dead/unhealthy or restarted, action taken (restarted, investigated, stopped, started), error or failure, degraded state, warning, threshold crossed, unexpected behavior, something was found or changed — the human should know this happened",
            "silent": "pure routine status with zero events: everything healthy, no actions taken, no events, no thresholds crossed, nothing found, no changes — an unremarkable periodic snapshot or an explicit 'all good, nothing to report'",
        },
    },
}
RANKS = ["k2", "deepseek", "luna", "terra"]
FORCE_TIERS = RANKS + ["glm"]

# ---------------------------------------------------------------- sessions
import hashlib
SESSIONS_FILE = ROOT / "sessions.json"
SESSION_CFG = CONFIG.get("session", {"ttl_hours": 4, "reeval_turns": 25})
SUBAGENT_MARKERS = ["cron job", "scheduled", "subagent", "worker", "delegated",
                    "you are running as"]
# MICRO-DELEGATION question: the criteria are written to defend against the
# failure mode observed in production: session-referential interjections
# ("what was your third recommendation?") LOOK self-contained to a classifier
# but require accumulated context. The "session" option explicitly wins ties:
# a missed downshift costs pennies; a context-free answer costs trust.
MICRO_QUESTIONS = {
    "delegate": {
        "type": "choice",
        "instructions": "This is a quick interjection inside an ongoing session that is otherwise doing substantive work. Can this specific message be fully and correctly handled by a small fast local model, without any context from the broader session?",
        "criteria": {
            "k2": "yes: fully self-contained — a quick fact, simple arithmetic, a greeting, a yes/no, a tiny rewrite, a lookup. No reference to anything said or done earlier in the conversation",
            "session": "no: it references the ongoing task or earlier messages (even indirectly — words like 'back to', 'your', 'that', 'it', 'our', 'the recommendation', 'as discussed'), needs the session's accumulated context, involves real reasoning/code/analysis, or is ambiguous. When in doubt, choose session",
        },
    },
}
_sessions = {}
try:
    _sessions = json.loads(SESSIONS_FILE.read_text())
except Exception:
    _sessions = {}


def _save_sessions():
    try:
        SESSIONS_FILE.write_text(json.dumps(_sessions))
    except Exception as e:
        log.warning(f"sessions save failed: {e}")


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

router_laya = Router(preload=["english"], device=os.environ.get("LAYA_DEVICE", "cuda"))


async def laya_decide(state: str):
    # Laya's predict() is a blocking forward pass. Agents fire concurrent
    # requests (subagent fan-outs, cron bursts), so running it in a thread keeps
    # the async event loop free so classifications happen in parallel, not
    # serialized behind each other. (Measured: 6 parallel classifications in
    # 1.8s wall vs ~11s serialized.)
    return await asyncio.to_thread(_laya_decide_sync, state)


def _laya_decide_sync(state: str):
    res = router_laya.predict(state, LAYA_QUESTIONS)
    a = res["answers"]
    t = a["tier"]
    tier_conf = t.get("answer_confidence") or t.get("probabilities", {}).get(t["choice"], 0)
    return {
        "tier": t["choice"],
        "tier_conf": tier_conf,
        "complexity": a["complexity"]["score"],
        "cx_conf": a["complexity"]["confidence"],
        "via": "laya",
    }


async def jev_decide(state: str):
    key = json.loads(AUTH_FILE.read_text())["vercel-ai-gateway"]["key"]
    r = await client.post(
        CONFIG["jev"]["url"],
        headers={"Authorization": f"Bearer {key}"},
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


def _micro_laya_sync(state: str):
    res = router_laya.predict(state, MICRO_QUESTIONS)
    a = res["answers"]["delegate"]
    conf = a.get("answer_confidence") or a.get("probabilities", {}).get(a["choice"], 0)
    return a["choice"], conf


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
    # Complexity escalation is deliberately asymmetric. Laya's score primitive is
    # its weakest (the model card admits ordinal scoring underperforms), and it
    # inflated trivial input ("Testing 123" scored 1.35/4, which under naive
    # absolute bands over-escalated to a paid cloud model). So thresholds are
    # RELATIVE to the tier Laya already chose, and escalation is the only
    # direction: worst case we slightly overpay; we never hand a task to a brain
    # that can't do it.
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


def _gate_laya_sync(state: str):
    """Sync wrapper: laya predict is CPU/GPU work, so run in a thread."""
    res = router_laya.predict(state, GATE_QUESTIONS)
    a = res["answers"]["notify"]
    conf = a.get("answer_confidence") or a.get("probabilities", {}).get(a["choice"], 0)
    return {"label": a["choice"], "conf": conf, "via": "laya"}


async def gate_laya(state: str):
    return await asyncio.to_thread(_gate_laya_sync, state)


async def gate_jev(state: str):
    key = json.loads(AUTH_FILE.read_text())["vercel-ai-gateway"]["key"]
    r = await client.post(
        CONFIG["jev"]["url"],
        headers={"Authorization": f"Bearer {key}"},
        json={
            "model": CONFIG["jev"]["model"],
            "state": state,
            "questions": {"notify": GATE_QUESTIONS["notify"]},
        },
        timeout=30,
    )
    a = r.json()["answers"]["notify"]
    return {"label": a["choice"], "conf": a["probabilities"][a["choice"]], "via": "jev"}


# GATE: watchdog alert triage (notify the human vs stay silent). The design
# invariant: a dead or confused gate may only ever over-notify, never
# under-notify: a redundant ping is noise, a swallowed outage is an incident.
#
# Layer 1: deterministic event markers. These words unambiguously mean
# "something happened" regardless of surrounding context ("back up, healthy
# now" after "Exited" must NOT classify as silent). Checked before the
# classifier so the free rule wins and the model only arbitrates genuinely
# ambiguous text. (In production: 69/69 notifies caught by this rule.)
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
    """Layer 2: model arbitration of ambiguous alerts, still fail-open.
    Silence must be earned with confidence. Anything else notifies.
    (Production: 210 checks, 0 missed events.)"""
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


TAG_RE = re.compile(r"@(k2|deepseek|glm|luna|terra)\b", re.IGNORECASE)
BRAIN_RE = re.compile(r"brain\s*[:=]\s*(k2|deepseek|glm|luna|terra)\b", re.IGNORECASE)


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


# ---------------------------------------------------------------- auth
def gateway_key() -> str:
    return json.loads(AUTH_FILE.read_text())["vercel-ai-gateway"]["key"]


def codex_auth():
    d = json.loads(AUTH_FILE.read_text())["openai-codex"]
    tok = d["access"]["token"] if isinstance(d.get("access"), dict) else d["access"]
    return tok, d.get("accountId")


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


def messages_to_responses(messages: list):
    """OpenAI chat messages -> codex responses-API instructions+input."""
    instructions = "\n".join(m["content"] for m in messages if m.get("role") == "system"
                             and isinstance(m.get("content"), str))
    input_items = []
    for m in messages:
        if m.get("role") == "system":
            continue
        c = m.get("content")
        if isinstance(c, str):
            input_items.append({"role": m["role"], "content": [{"type": "input_text", "text": c}]})
        elif isinstance(c, list):
            items = [{"type": "input_text", "text": p["text"]} for p in c
                     if isinstance(p, dict) and p.get("type") in ("text", "input_text")]
            input_items.append({"role": m["role"], "content": items})
    return instructions or None, input_items


# ---------------------------------------------------------------- app
app = FastAPI()
client = httpx.AsyncClient(timeout=httpx.Timeout(600, connect=15))


@app.get("/health")
async def health():
    return {"ok": True, "backends": list(CONFIG["backends"])}


def _append_stat(record: dict):
    try:
        with open(STATS_FILE, "a") as f:
            f.write(json.dumps(record) + "\n")
    except Exception as e:
        log.warning(f"stats append failed: {e}")


@app.get("/v1/recent")  # last N decisions, newest first (rich records)
async def recent(n: int = 10):
    n = max(1, min(n, 100))
    try:
        with open(STATS_FILE) as f:
            rows = [json.loads(l) for l in f if l.strip()]
    except FileNotFoundError:
        rows = []
    return {"decisions": list(reversed(rows[-n:]))}


@app.get("/v1/stats")  # persistent usage stats (survives router.log rotation)
async def stats():
    rows = []
    try:
        with open(STATS_FILE) as f:
            for line in f:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
    except FileNotFoundError:
        pass
    routes = [r for r in rows if r.get("type", "route") == "route"]
    gates = [r for r in rows if r.get("type") == "gate"]
    def tally(items, key):
        t = {}
        for r in items:
            k = r.get(key)
            t[k] = t.get(k, 0) + 1
        return dict(sorted(t.items(), key=lambda kv: -kv[1]))
    by_day = {}
    tokens_by_tier = {}
    for r in routes:
        day = r.get("ts", "?")[:10]
        by_day[day] = by_day.get(day, 0) + 1
        t = r.get("tier", "?")
        tokens_by_tier[t] = tokens_by_tier.get(t, 0) + (r.get("est_tokens") or 0)
    prices = CONFIG.get("stats_prices_per_mtok_input", {})
    cost = {t: round(n / 1_000_000 * prices.get(t, 0), 4) for t, n in tokens_by_tier.items()}
    return {
        "total_routes": len(routes),
        "total_gate_checks": len(gates),
        "first": rows[0].get("ts") if rows else None,
        "last": rows[-1].get("ts") if rows else None,
        "by_tier": tally(routes, "tier"),
        "by_via": tally(routes, "via"),
        "by_day": dict(sorted(by_day.items())),
        "escalations": sum(1 for r in routes if r.get("escalated_by")),
        "k2_overflows_rescued": sum(1 for r in rows if r.get("overflow")),
        "gate": {"total": len(gates),
                 "by_label": tally(gates, "label"),
                 "by_via": tally(gates, "via"),
                 "by_rule": sum(1 for g in gates if g.get("via") == "rule")},
        "est_input_tokens_by_tier": tokens_by_tier,
        "est_input_cost_usd_by_tier": {t: round(n / 1_000_000 * prices.get(t, 0), 4)
                                        for t, n in tokens_by_tier.items()},
        "est_total_cost_usd": round(sum(n / 1_000_000 * prices.get(t, 0)
                                          for t, n in tokens_by_tier.items()), 4),
    }


@app.post("/v1/routing")  # dry run: decision only, no backend call
async def routing_probe(req: Request):
    body = await req.json()
    state = body.get("state") or state_from_messages(body.get("messages", []))
    tag = detect_tag(body.get("messages", []))
    asked = body.get("model")
    force = asked if asked in RANKS else tag
    est = (len(json.dumps(body.get("messages", []))) +
           len(json.dumps(body.get("tools", [])))) * 10 // 44
    d = await decide(state, force=force, est_tokens=est)
    if d["tier"] == "k2" and est > 30000:
        d["tier"] = "deepseek"
        d["escalated_by"] = "length"
        d["forced_k2_overflow"] = True
    b = CONFIG["backends"][d["tier"]]
    return {"tier": d["tier"], "backend_model": b["model"], "desc": b["desc"],
            "confidence": d["tier_conf"], "via": d["via"], "complexity": d.get("complexity")}


@app.post("/v1/gate")  # alert gate: classify a message as notify / silent
async def gate(req: Request):
    body = await req.json()
    state = body.get("text") or body.get("state") or ""
    if not state.strip():
        return {"label": "silent", "confidence": 1.0, "via": "empty", "notify": False}
    t0 = time.time()
    d = await gate_decide(state)
    ms = int((time.time() - t0) * 1000)
    log.info(f"gate -> {d['label']:<7} conf={d['conf']:.2f} via={d['via']} classify={ms}ms")
    try:
        record = {"type": "gate", "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                  "label": d["label"], "conf": round(d["conf"], 3), "via": d["via"],
                  "markers": d.get("markers", []), "classify_ms": ms,
                  "state_excerpt": state[:120]}
        with open(STATS_FILE, "a") as f:
            f.write(json.dumps(record) + "\n")
    except Exception as e:
        log.warning(f"gate stats append failed: {e}")
    return {"label": d["label"], "confidence": round(d["conf"], 3), "via": d["via"],
            "notify": d["label"] == "notify"}


@app.post("/v1/chat/completions")
async def chat(req: Request):
    body = await req.json()
    (ROOT / "last-request.json").write_text(json.dumps(body)[:200000])
    asked_model = body.get("model", "auto")
    tag = detect_tag(body.get("messages", []))
    force = asked_model if asked_model in FORCE_TIERS else tag
    state = state_from_messages(body.get("messages", []))
    ser = json.dumps(body.get("messages", [])) + json.dumps(body.get("tools", []))
    est_tokens = len(ser) * 10 // 44  # measured ~4.4 chars/token on pi payloads
    t0 = time.time()

    # ---- SESSION PINNING: classify once per session, not per turn ----
    # Per-message routing made agent sessions incoherent: a 40-turn bug-fix
    # bounced between three models, each losing the thread of its own work,
    # while paying ~2s of judge overhead per turn (~80s/session). Instead:
    # classify ONCE (first turn), pin the tier for the whole session.
    sess_key = session_key(body)
    now = time.time()
    sess = _sessions.get(sess_key) if sess_key else None
    if sess and now - sess.get("ts", 0) > SESSION_CFG.get("ttl_hours", 4) * 3600:
        sess = None
    d = None
    if not force and sess:
        # Pinned turn: skip classification entirely. This is where the session
        # coherence pays off: the same model that started the task finishes it.
        tier = sess["tier"]
        sess["turns"] += 1
        sess["since_reeval"] += 1
        sess["ts"] = now
        d = {"tier": tier, "via": "session", "tier_conf": sess.get("conf", 1.0),
             "complexity": None, "pinned": True, "session": sess_key[:8],
             "turn": sess["turns"], "kind": sess.get("kind", "main")}
        # MICRO-DELEGATION: mid-session the human asks something self-contained
        # ("what's 45*12?" from the car). Running it on the session's frontier
        # brain wastes capability; running the whole session on a small model
        # loses capability. So: cheap single-question check, and if Laya is
        # CONFIDENT the interjection is fully self-contained, hand just this
        # one turn to the local model. Session-referential turns ("what was
        # your third recommendation?") stay home. They need the context.
        if sess.get("kind") == "main" and tier in ("deepseek", "luna", "terra"):
            md = await micro_delegate(body, tier)
            if md:
                tier = d["tier"] = md["tier"]
                d["via"] = md["via"]
                d["tier_conf"] = md["conf"]
                b = CONFIG["backends"][tier]
                log.info(f"micro-delegated turn to k2 (home={sess['tier']})")
        if sess["since_reeval"] >= SESSION_CFG.get("reeval_turns", 25):
            # periodic re-check (promote-only: never demote mid-task)
            log.info(f"session {sess_key[:8]} re-eval after {sess['since_reeval']} turns")
            nd = await decide(state, est_tokens=est_tokens)
            new_rank, old_rank = RANKS.index(nd["tier"]), RANKS.index(tier)
            if new_rank > old_rank:
                tier = nd["tier"]
                d.update(tier=tier, via="session-reeval", conf=nd["tier_conf"],
                         complexity=nd.get("complexity"))
            sess["tier"] = tier
            sess["since_reeval"] = 0
            _save_sessions()
    elif not force:
        # first turn of a session: full classification, then pin.
        # Explicit brain: directive in the system prompt, or name-based rules,
        # outrun classification. Some crons need a bigger brain by design.
        brain = declared_brain(body) or rule_brain(body)
        if brain:
            d = {"tier": brain, "via": "declared", "tier_conf": 1.0,
                 "complexity": None, "pinned": True}
            log.info(f"session brain declared: {brain}")
        else:
            d = await decide(state, est_tokens=est_tokens)
        if sess_key:
            sysprompt = next((str(m.get("content","")) for m in body.get("messages",[])
                              if m.get("role") == "system"), "")
            low = sysprompt.lower()
            kind = "subagent" if any(mk in low for mk in SUBAGENT_MARKERS) else "main"
            if kind == "main" and d["tier"] == "k2" and not brain:
                # main sessions never pin k2: home floor is deepseek
                d["tier"] = "deepseek"
                d["escalated_by"] = "main-session-floor"
            _sessions[sess_key] = {"tier": d["tier"], "ts": now, "turns": 1,
                                   "since_reeval": 0, "conf": d["tier_conf"],
                                   "overflows": 0, "kind": kind, "brain": brain}
            _save_sessions()
        d["pinned"] = True
        d["session"] = sess_key[:8] if sess_key else None
    if d is None:  # forced tier this turn (client pin or @tag), no session write
        d = await decide(state, force=force, est_tokens=est_tokens)
    tier = d["tier"]
    b = CONFIG["backends"][tier]
    # OVERFLOW RESCUE: the local model's context is finite (32K). Full agentic
    # payloads (system prompt + 22 tool schemas + history) can exceed it, and
    # the client's own truncation doesn't count tool schemas. Instead of a 400,
    # rescue to the next tier; after repeated rescues, promote the session so
    # it stops retrying a brain that can't hold it.
    K2_CTX = 32768
    if tier == "k2":
        if est_tokens > 30000:
            log.warning(f"k2 overflow rescue: est prompt {est_tokens} > 30000 -> deepseek")
            tier = d["tier"] = "deepseek"
            b = CONFIG["backends"][tier]
            d["escalated_by"] = "length"
            d["forced_k2_overflow"] = True
            if sess and sess.get("tier") == "k2":
                sess["overflows"] = sess.get("overflows", 0) + 1
                if sess["overflows"] >= 2:  # this session simply doesn't fit k2
                    sess["tier"] = "deepseek"
                    log.info(f"session {sess_key[:8]} promoted to deepseek after {sess['overflows']} overflows")
                    _save_sessions()
        else:
            allowed_out = max(1024, K2_CTX - est_tokens - 512)
            if body.get("max_tokens", 0) > allowed_out:
                body = {**body, "max_tokens": allowed_out}
    ms = int((time.time() - t0) * 1000)
    log.info(f"route -> {tier:<8} via={d['via']:<7} conf={d['tier_conf']:.2f} "
             f"cx={d.get('complexity')} asked={asked_model} tag={tag} classify={ms}ms sess={sess_key[:8] if sess_key else '-'} pin={d.get('pinned')}")
    try:
        record = {"type": "route", "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                  "tier": tier, "via": d["via"],
                  "conf": round(d["tier_conf"], 2),
                  "complexity": d.get("complexity"),
                  "est_tokens": est_tokens, "asked": asked_model,
                  "tag": tag, "classify_ms": ms,
                  "escalated_by": d.get("escalated_by"),
                  "overflow": bool(d.get("forced_k2_overflow")),
                  "session": sess_key[:8] if sess_key else None,
                  "pinned": bool(d.get("pinned")), "turn": d.get("turn"),
                  "laya": {"tier": d.get("tier") if d.get("via") == "laya" else d.get("laya_tier"),
                            "tier_probs": d.get("tier_probs"),
                            "stakes": d.get("stakes"), "stakes_probs": d.get("stakes_probs")},
                  "jev": d.get("jev"),
                  "state_excerpt": state[:200]}
        with open(STATS_FILE, "a") as f:
            f.write(json.dumps(record) + "\n")
    except Exception as e:
        log.warning(f"stats append failed: {e}")
    rid = f"rtr_{int(time.time()*1000)}"

    if body.get("stream"):
        return StreamingResponse(stream_openai(body, tier, b, d, rid),
                                 media_type="text/event-stream",
                                 headers={"X-Router-Tier": tier, "X-Router-Via": d["via"],
                                          "X-Router-Confidence": f"{d['tier_conf']:.2f}"})
    out = await complete_openai(body, tier, b, rid)
    out["router"] = {"tier": tier, "backend_model": b["model"], "via": d["via"],
                     "confidence": round(d["tier_conf"], 2), "complexity": d.get("complexity"),
                     "classify_ms": ms}
    return JSONResponse(out, headers={"X-Router-Tier": tier, "X-Router-Via": d["via"],
                                      "X-Router-Confidence": f"{d['tier_conf']:.2f}"})


# ---- openai-compatible backends (k2, deepseek)
async def complete_openai(body, tier, b, rid):
    payload = {k: v for k, v in body.items() if k not in ("model", "stream")}
    tried = []
    candidates = [tier] + CONFIG["backends"].get(tier, {}).get("fallbacks", [])
    for candidate in candidates:
        cb = CONFIG["backends"].get(candidate)
        if not cb or cb.get("type") != "openai":
            continue
        tried.append(candidate)
        payload["model"] = cb["model"]
        try:
            r = await client.post(f"{cb['base_url']}/chat/completions", json=payload,
                                  headers={"Authorization": f"Bearer {_key_for(cb)}"})
        except Exception as e:
            log.warning(f"backend {candidate} request failed: {e}")
            continue
        if r.status_code != 200:
            log.warning(f"backend {candidate} {r.status_code}: {r.text[:500]}")
            continue
        try:
            out = r.json()
        except Exception:
            out = {"error": r.text[:500]}
        out["id"] = rid
        if candidate != tier:
            out["router_fallback"] = {"from": tier, "to": candidate, "tried": tried}
        return out
    (ROOT / "last-failed-request.json").write_text(json.dumps(body)[:200000])
    return {"error": {"message": f"all backends failed starting at {tier}", "tried": tried}}


def _key_for(b):
    if b.get("api_key"):
        return b["api_key"]
    if b.get("auth") == "gateway":
        return gateway_key()
    return ""


async def stream_openai(body, tier, b, d, rid):
    """Yield an OpenAI-compatible SSE stream; falls back across openai + codex backends."""
    candidates = [tier] + CONFIG["backends"].get(tier, {}).get("fallbacks", [])
    tried = []
    for candidate in candidates:
        cb = CONFIG["backends"].get(candidate)
        if not cb:
            continue
        tried.append(candidate)
        if cb["type"] == "openai":
            payload = {k: v for k, v in body.items() if k not in ("model",)}
            payload.setdefault("stream", True)
            payload["model"] = cb["model"]
            try:
                async with client.stream("POST", f"{cb['base_url']}/chat/completions", json=payload,
                                         headers={"Authorization": f"Bearer {_key_for(cb)}"}) as r:
                    if r.status_code != 200:
                        detail = (await r.aread())[:500].decode(errors="replace")
                        log.warning(f"backend {candidate} {r.status_code}: {detail}")
                        continue
                    first = {"id": rid, "object": "chat.completion.chunk", "model": cb["model"],
                             "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]}
                    yield f"data: {json.dumps(first)}\n\n"
                    if candidate != tier:
                        log.warning(f"stream fallback {tier} -> {candidate}; tried={tried}")
                    async for chunk in r.aiter_bytes():
                        yield chunk
                    return
            except Exception as e:
                log.warning(f"backend {candidate} stream failed: {e}")
                continue
        elif cb["type"] == "codex":
            try:
                tok, acct = codex_auth()
            except Exception as e:
                log.warning(f"codex {candidate} auth failed: {e}")
                continue
            instructions, inp = messages_to_responses(body.get("messages", []))
            payload = {"model": cb["model"], "input": inp, "store": False, "stream": True}
            if instructions:
                payload["instructions"] = instructions
            headers = {"Authorization": f"Bearer {tok}", "chatgpt-account-id": acct or "",
                       "OpenAI-Beta": "responses=experimental", "originator": "pi",
                       "Content-Type": "application/json", "Accept": "text/event-stream"}
            try:
                async with client.stream("POST", "https://chatgpt.com/backend-api/codex/responses",
                                         json=payload, headers=headers) as r:
                    if r.status_code != 200:
                        detail = (await r.aread())[:500].decode(errors="replace")
                        log.warning(f"codex {candidate} {r.status_code}: {detail}")
                        continue
                    first = {"id": rid, "object": "chat.completion.chunk", "model": cb["model"],
                             "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]}
                    yield f"data: {json.dumps(first)}\n\n"
                    if candidate != tier:
                        log.warning(f"stream fallback {tier} -> {candidate}; tried={tried}")
                    finished = False
                    async for line in r.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        try:
                            ev = json.loads(line[6:])
                        except Exception:
                            continue
                        t = ev.get("type")
                        if t == "response.output_text.delta":
                            chunk = {"id": rid, "object": "chat.completion.chunk", "model": cb["model"],
                                     "choices": [{"index": 0, "delta": {"content": ev.get("delta", "")}, "finish_reason": None}]}
                            yield f"data: {json.dumps(chunk)}\n\n"
                        elif t in ("response.failed", "error", "response.incomplete"):
                            detail = json.dumps(ev)[:400]
                            log.warning(f"codex stream {t}: {detail}")
                            yield f"data: {json.dumps({'error': {'message': f'codex stream {t}', 'detail': detail}})}\n\n"
                            yield "data: [DONE]\n\n"
                            return
                        elif t == "response.completed":
                            finished = True
                            done = {"id": rid, "object": "chat.completion.chunk", "model": cb["model"],
                                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
                            yield f"data: {json.dumps(done)}\n\n"
                    if not finished:
                        done = {"id": rid, "object": "chat.completion.chunk", "model": cb["model"],
                                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
                        yield f"data: {json.dumps(done)}\n\n"
                    yield "data: [DONE]\n\n"
                    return
            except Exception as e:
                log.warning(f"codex {candidate} stream failed: {e}")
                continue
    (ROOT / "last-failed-request.json").write_text(json.dumps(body)[:200000])
    err = {"error": {"message": f"all streaming backends failed starting at {tier}", "tried": tried}}
    yield f"data: {json.dumps(err)}\n\n"
    yield "data: [DONE]\n\n"