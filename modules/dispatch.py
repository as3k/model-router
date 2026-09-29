"""OpenAI-compatible backend dispatch: non-streaming and SSE streaming with
fallback across openai and codex backends (k2 -> deepseek -> luna -> terra).

Also hosts the per-backend circuit breaker (skip-first deprioritization of
recently failing backends, no probes) and the k2 single-slot priority gate
(main-session turns dispatch before subagent/cron turns)."""
import asyncio
import json
import time
from collections import deque

from modules.classify import RANKS, last_user_msg, router_laya
from modules.settings import CONFIG, ROOT, log, gateway_key, codex_auth

client = None  # shared httpx AsyncClient, set by router.py


# ------------------------------------------------------------ circuit breaker
# In-memory per-backend failure memory: {backend: {last_fail_ts, fail_streak}}.
# No background probes: cooldown expiry IS the natural retry.
_failures: dict = {}
COOLDOWN_S = 120  # backends with a failure inside this window move to the end


def _record_failure(backend: str):
    f = _failures.setdefault(backend, {"last_fail_ts": 0.0, "fail_streak": 0})
    f["last_fail_ts"] = time.time()
    f["fail_streak"] += 1


def _ordered_candidates(candidates: list) -> list:
    """Skip-first, not never-try: recently-failed backends go to the END of the
    candidate list (still tried if everything ahead of them is down)."""
    now = time.time()
    healthy, sick = [], []
    for c in candidates:
        f = _failures.get(c)
        if f and now - f["last_fail_ts"] < COOLDOWN_S:
            sick.append(c)
            log.warning(f"circuit: {c} deprioritized for {COOLDOWN_S - (now - f['last_fail_ts']):.0f}s")
        else:
            healthy.append(c)
    return healthy + sick


# ------------------------------------------------------------ k2 priority gate
# k2 is a single-decode-slot backend; concurrent requests queue inside
# llama-server in arrival order. This gate reorders the queue: main-session
# turns (priority 0) dispatch before subagent/cron turns (priority 1). Two FIFO
# wait lists and a tiny scheduler; a steady stream of main turns can starve
# priority 1 while it lasts. Accepted: k2 turns are short and cheap.
class _K2Gate:
    def __init__(self, slots: int = 1):
        self._slots = slots
        self._in_use = 0
        self._waiters = {0: deque(), 1: deque()}  # priority -> waiter events (FIFO)
        self._lock = asyncio.Lock()

    async def acquire(self, priority: int):
        ev = asyncio.Event()
        async with self._lock:
            self._waiters[priority].append(ev)
            self._grant()
        await ev.wait()

    async def release(self):
        async with self._lock:
            self._in_use -= 1
            self._grant()

    def _grant(self):
        while self._in_use < self._slots and (self._waiters[0] or self._waiters[1]):
            q = self._waiters[0] if self._waiters[0] else self._waiters[1]
            q.popleft().set()
            self._in_use += 1


_gate = _K2Gate()


class _K2GateCtx:
    def __init__(self, gate: _K2Gate, priority: int):
        self._gate, self._priority = gate, priority

    async def __aenter__(self):
        await self._gate.acquire(self._priority)
        return self

    async def __aexit__(self, *exc):
        await self._gate.release()
        return False


def k2_gate(priority: int = 1):
    """async with k2_gate(0): ...  priority 0 (main) beats 1 (subagent)."""
    return _K2GateCtx(_gate, priority)


# ------------------------------------------------------------ quality check (v1)
# v1 covers non-streaming responses only; streaming turns are not checked yet.
QUALITY_QUESTION = {
    "quality": {
        "type": "choice",
        "instructions": "Did this response actually address the request, or is it a refusal, confusion, or off-task?",
        "criteria": {
            "addressed": "the response engages with the actual request: it answers, explains, or does the work asked",
            "off_task": "the response is a refusal, a confusion, a dodge, unrelated output, or it misunderstood the request",
        },
    },
}


async def _quality_escalation(body: dict, out: dict, served_tier: str) -> str | None:
    """Laya check on (last user message + response head 1500 chars). Returns the
    next tier up when the answer is off-task (conf >= 0.6) and the served tier
    ranks below terra; None otherwise. Caller retries the turn there, once."""
    if served_tier in RANKS:
        rk = RANKS.index(served_tier)
    else:  # glm and other non-RANKS backends use their config rank
        rk = CONFIG["backends"].get(served_tier, {}).get("rank", 0)
    if rk >= 3 or rk + 1 >= len(RANKS):
        return None  # never auto-escalate terra
    try:
        content = (out.get("choices") or [{}])[0].get("message", {}).get("content", "")
    except Exception:
        return None
    if not isinstance(content, str) or not content.strip():
        return None
    state = f"{(last_user_msg(body) or '')[:2000]}\n\n{content[:1500]}"
    try:
        def _ask():
            a = router_laya.predict(state, QUALITY_QUESTION)["answers"]["quality"]
            conf = a.get("answer_confidence") or a.get("probabilities", {}).get(a["choice"], 0)
            return a["choice"], conf
        choice, conf = await asyncio.to_thread(_ask)
    except Exception as e:
        log.warning(f"quality check failed: {e}")
        return None
    if choice == "off_task" and conf >= 0.6:
        return RANKS[rk + 1]
    return None


# ---- openai-compatible backends (k2, deepseek)
async def complete_openai(body, tier, b, rid, k2_priority: int = 1, _escalating: bool = False):
    payload = {k: v for k, v in body.items() if k not in ("model", "stream")}
    tried = []
    candidates = _ordered_candidates([tier] + CONFIG["backends"].get(tier, {}).get("fallbacks", []))
    for candidate in candidates:
        cb = CONFIG["backends"].get(candidate)
        if not cb or cb.get("type") != "openai":
            continue
        tried.append(candidate)
        payload["model"] = cb["model"]
        queue_ms = 0
        try:
            if candidate == "k2":
                tq = time.time()
                async with k2_gate(k2_priority):
                    queue_ms = int((time.time() - tq) * 1000)
                    r = await client.post(f"{cb['base_url']}/chat/completions", json=payload,
                                          headers={"Authorization": f"Bearer {_key_for(cb)}"})
            else:
                r = await client.post(f"{cb['base_url']}/chat/completions", json=payload,
                                      headers={"Authorization": f"Bearer {_key_for(cb)}"})
        except Exception as e:
            _record_failure(candidate)
            log.warning(f"backend {candidate} request failed: {e}")
            continue
        if r.status_code != 200:
            _record_failure(candidate)
            log.warning(f"backend {candidate} {r.status_code}: {r.text[:500]}")
            continue
        try:
            out = r.json()
        except Exception:
            out = {"error": r.text[:500]}
        out["id"] = rid
        out["_queue_ms"] = queue_ms
        if candidate != tier:
            out["router_fallback"] = {"from": tier, "to": candidate, "tried": tried}
        # quality auto-escalation (v1): non-streaming only, once per turn.
        # A successful response from a sub-terra tier that laya judges off-task
        # (conf >= 0.6) is retried once on the next tier up.
        if not _escalating:
            nxt = await _quality_escalation(body, out, candidate)
            if nxt:
                nb = CONFIG["backends"][nxt]
                log.warning(f"quality: {candidate} answer off-task -> retrying once on {nxt}")
                retry = await complete_openai(body, nxt, nb, rid,
                                              k2_priority=k2_priority, _escalating=True)
                retry["_quality_escalation"] = {"from": candidate, "to": nxt}
                return retry
        return out
    (ROOT / "last-failed-request.json").write_text(json.dumps(body)[:200000])
    return {"error": {"message": f"all backends failed starting at {tier}", "tried": tried}}


def _key_for(b):
    if b.get("api_key"):
        return b["api_key"]
    if b.get("auth") == "gateway":
        return gateway_key()
    return ""


async def stream_openai(body, tier, b, d, rid, k2_priority: int = 1, queue_holder: dict | None = None):
    """Yield an OpenAI-compatible SSE stream; falls back across openai + codex backends."""
    candidates = _ordered_candidates([tier] + CONFIG["backends"].get(tier, {}).get("fallbacks", []))
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
            headers = {"Authorization": f"Bearer {_key_for(cb)}"}
            try:
                if candidate == "k2":  # hold the k2 decode slot for the whole stream
                    tq = time.time()
                    async with k2_gate(k2_priority):
                        if queue_holder is not None:
                            queue_holder["queue_ms"] = int((time.time() - tq) * 1000)
                        async with client.stream("POST", f"{cb['base_url']}/chat/completions",
                                                 json=payload, headers=headers) as r:
                            if r.status_code != 200:
                                _record_failure(candidate)
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
                else:
                    async with client.stream("POST", f"{cb['base_url']}/chat/completions",
                                             json=payload, headers=headers) as r:
                        if r.status_code != 200:
                            _record_failure(candidate)
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
                _record_failure(candidate)
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
                        _record_failure(candidate)
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
                _record_failure(candidate)
                log.warning(f"codex {candidate} stream failed: {e}")
                continue
    (ROOT / "last-failed-request.json").write_text(json.dumps(body)[:200000])
    err = {"error": {"message": f"all streaming backends failed starting at {tier}", "tried": tried}}
    yield f"data: {json.dumps(err)}\n\n"
    yield "data: [DONE]\n\n"


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