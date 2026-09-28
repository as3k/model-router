"""OpenAI-compatible backend dispatch: non-streaming and SSE streaming with
fallback across openai and codex backends (k2 -> deepseek -> luna -> terra)."""
import json

from modules.settings import CONFIG, ROOT, log, gateway_key, codex_auth

client = None  # shared httpx AsyncClient, set by router.py


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