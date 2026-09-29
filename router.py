#!/usr/bin/env python3
"""Laya/Jev-fronted model router.

OpenAI-compatible proxy: POST /v1/chat/completions.
Classifies each request with Laya (local, ~35ms), falls back to Jev on
low confidence, then forwards to the backend matching the required
specificity: k2 -> deepseek -> luna -> terra.

Routing decision is exposed via X-Router-* response headers, the
`router` field of the response body, and ~/ml/router/router.log.

Composition: classify (deciders + policy), sessions (registry), dispatch
(backend calls), stats_store (stats.jsonl). See those modules.
"""
import asyncio
import json
import time

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from modules import classify, compact, dispatch, spend, stats_store
from modules.spend import tracker as spend_tracker
from modules.classify import (RANKS, FORCE_TIERS, SESSION_CFG, SUBAGENT_MARKERS,
                      decide, gate_decide, detect_tag, declared_brain,
                      micro_delegate, rule_brain, session_key, state_from_messages)
from modules.dispatch import complete_openai, stream_openai
from modules.sessions import _sessions, save as save_sessions, expired as session_expired
from modules.settings import CONFIG, ROOT, log

# ---------------------------------------------------------------- app
app = FastAPI()
client = httpx.AsyncClient(timeout=httpx.Timeout(600, connect=15))
classify.client = client
dispatch.client = client
compact.client = client
spend.client = client


# spend soft-limit notification: fire-and-forget POST to our own alert gate
# (the gate notifies; that is intended).
async def _spend_notify(text: str):
    try:
        await spend.client.post("http://127.0.0.1:8090/v1/gate",
                                json={"text": text}, timeout=10)
    except Exception as e:
        log.warning(f"spend gate notify failed: {e}")


def _spend_notify_task(text: str):
    asyncio.create_task(_spend_notify(text))


@app.get("/health")
async def health():
    return {"ok": True, "backends": list(CONFIG["backends"])}


@app.get("/v1/recent")  # last N decisions, newest first (rich records)
async def recent(n: int = 10):
    return {"decisions": stats_store.recent_rows(n)}


@app.get("/v1/stats")  # persistent usage stats (survives router.log rotation)
async def stats():
    return stats_store.aggregate()


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
    record = {"type": "gate", "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
              "label": d["label"], "conf": round(d["conf"], 3), "via": d["via"],
              "markers": d.get("markers", []), "classify_ms": ms,
              "state_excerpt": state[:120]}
    stats_store.append_route(record)
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

    # ---- session pinning: classify ONCE per session, not per turn ----
    sess_key = session_key(body)
    now = time.time()
    sess = _sessions.get(sess_key) if sess_key else None
    if sess and session_expired(sess, SESSION_CFG.get("ttl_hours", 4)):
        sess = None
    d = None
    if not force and sess:
        # pinned: zero classification cost this turn
        tier = sess["serving_tier"]
        sess["turns"] += 1
        sess["since_reeval"] += 1
        sess["ts"] = now
        save_sessions()  # keep the persisted registry current per turn
        d = {"tier": tier, "via": "session", "tier_conf": sess.get("conf", 1.0),
             "complexity": None, "pinned": True, "session": sess_key[:8],
             "turn": sess["turns"], "kind": sess.get("kind", "main")}
        # micro-delegation: a short, self-contained interjection inside a main
        # session can be handled one-off by k2 (home tier is untouched)
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
            sess["serving_tier"] = tier
            sess["since_reeval"] = 0
            save_sessions()
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
            d["kind"] = kind
            _sessions[sess_key] = {"serving_tier": d["tier"], "ts": now, "turns": 1,
                                   "since_reeval": 0, "conf": d["tier_conf"],
                                   "overflows": 0, "kind": kind, "declared_tier": brain}
            save_sessions()
        d["pinned"] = True
        d["session"] = sess_key[:8] if sess_key else None
    if d is None:  # forced tier this turn (client pin or @tag), no session write
        d = await decide(state, force=force, est_tokens=est_tokens)
    tier = d["tier"]
    b = CONFIG["backends"][tier]
    # k2 fit: ctx 32768. Prompt alone must fit; output gets the remainder.
    # Overflow first tries COMPACTION (summarize old turns, keep recent ones
    # verbatim) so declared-k2 sessions stay on k2; rescue to deepseek only
    # when compaction isn't possible or the compacted payload still overflows.
    K2_CTX = 32768
    if tier == "k2":
        if est_tokens > 30000 and sess_key and sess:
            from modules import compact
            try:
                c = await compact.maybe_compact(client, body, sess, sess_key)
            except Exception as e:
                log.warning(f"compaction failed: {e}")
                c = None
            if c:
                body = {**body, "messages": c["messages"]}
                est_tokens = (len(json.dumps(c["messages"])) +
                              len(json.dumps(body.get("tools", [])))) * 10 // 44
                d["compacted"] = True
                d["compact_span"] = c["span_messages"]
                save_sessions()  # persist the compact state (summary + prefix hash)
                log.info(f"compacted session {sess_key[:8]}: summarized "
                         f"{c['span_messages']} messages, new est {est_tokens}")
        if tier == "k2" and est_tokens > 30000:
            if spend_tracker.hard_active:
                # spend guard hard limit: no deepseek rescue — stay on k2, clamp
                # the output budget so the prompt still fits. Compaction (above)
                # was already attempted; this is the k2-with-compaction fallback.
                log.warning(f"k2 overflow (est {est_tokens}) held on k2: spend guard hard limit active")
                allowed_out = max(1024, K2_CTX - est_tokens - 512)
                if body.get("max_tokens", 0) > allowed_out:
                    body = {**body, "max_tokens": allowed_out}
            else:
                log.warning(f"k2 overflow rescue: est prompt {est_tokens} > 30000 -> deepseek")
                tier = d["tier"] = "deepseek"
                b = CONFIG["backends"][tier]
                d["escalated_by"] = "length"
                d["forced_k2_overflow"] = True
                if sess and sess.get("serving_tier") == "k2":
                    sess["overflows"] = sess.get("overflows", 0) + 1
                    if sess["overflows"] >= 2:  # this session simply doesn't fit k2
                        sess["serving_tier"] = "deepseek"
                        log.info(f"session {sess_key[:8]} promoted to deepseek after {sess['overflows']} overflows")
                        save_sessions()
        else:
            allowed_out = max(1024, K2_CTX - est_tokens - 512)
            if body.get("max_tokens", 0) > allowed_out:
                body = {**body, "max_tokens": allowed_out}
    # ---- spend guardrails: daily counter, soft/hard limits (reset at midnight) ----
    spend_ev = spend_tracker.note_usage(tier, est_tokens)
    if spend_ev["soft"]:
        msg = f"model router spend crossed soft limit: ${spend_tracker.total:.2f} today"
        log.warning(msg)
        _spend_notify_task(msg)
    if spend_ev["hard"]:
        log.warning(f"spend guard: hard limit ${spend_tracker.hard_limit():.2f} crossed — "
                    f"degraded mode until midnight (jev tiebreaker skipped, k2 overflows stay on k2)")

    # k2 slot priority: main-session turns (0) dispatch before subagent turns (1)
    k2_prio = 0 if d.get("kind") == "main" else 1
    ms = int((time.time() - t0) * 1000)
    log.info(f"route -> {tier:<8} via={d['via']:<7} conf={d['tier_conf']:.2f} "
             f"cx={d.get('complexity')} asked={asked_model} tag={tag} classify={ms}ms sess={sess_key[:8] if sess_key else '-'} pin={d.get('pinned')}")
    record = {"type": "route", "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
              "tier": tier, "via": d["via"],
              "conf": round(d["tier_conf"], 2),
              "complexity": d.get("complexity"),
              "est_tokens": est_tokens, "asked": asked_model,
              "tag": tag, "classify_ms": ms, "queue_ms": 0,
              "escalated_by": d.get("escalated_by"),
              "overflow": bool(d.get("forced_k2_overflow")),
                  "compacted": bool(d.get("compacted")), "compact_span": d.get("compact_span"),
              "session": sess_key[:8] if sess_key else None,
              "pinned": bool(d.get("pinned")), "turn": d.get("turn"),
              "laya": {"tier": d.get("tier") if d.get("via") == "laya" else d.get("laya_tier"),
                        "tier_probs": d.get("tier_probs"),
                        "stakes": d.get("stakes"), "stakes_probs": d.get("stakes_probs")},
              "jev": d.get("jev"),
              "state_excerpt": state[:200]}
    rid = f"rtr_{int(time.time()*1000)}"

    if body.get("stream"):
        queue_holder: dict = {}
        sgen = stream_openai(body, tier, b, d, rid, k2_priority=k2_prio,
                             queue_holder=queue_holder)

        async def _stream_then_record():
            try:
                async for chunk in sgen:
                    yield chunk
            finally:
                # k2 queue wait is only known once the stream actually dispatched;
                # commit the route record when the stream ends (or is closed).
                record["queue_ms"] = queue_holder.get("queue_ms", 0)
                stats_store.append_route(record)

        return StreamingResponse(_stream_then_record(),
                                 media_type="text/event-stream",
                                 headers={"X-Router-Tier": tier, "X-Router-Via": d["via"],
                                          "X-Router-Confidence": f"{d['tier_conf']:.2f}"})
    out = await complete_openai(body, tier, b, rid, k2_priority=k2_prio)
    qe = out.pop("_quality_escalation", None)
    record["queue_ms"] = out.pop("_queue_ms", 0)
    if qe:
        record["quality_escalated"] = qe
    stats_store.append_route(record)
    out["router"] = {"tier": tier, "backend_model": b["model"], "via": d["via"],
                     "confidence": round(d["tier_conf"], 2), "complexity": d.get("complexity"),
                     "classify_ms": ms, "compacted": bool(d.get("compacted")),
                     "compact_span": d.get("compact_span")}
    if qe:
        out["router"]["quality_escalated"] = qe
    return JSONResponse(out, headers={"X-Router-Tier": tier, "X-Router-Via": d["via"],
                                      "X-Router-Confidence": f"{d['tier_conf']:.2f}"})