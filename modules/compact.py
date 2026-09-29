"""Session compaction: keeps large sessions usable on small-context models.

Trigger: the serving tier is k2 and the estimated payload exceeds k2's
comfortable fit. Instead of rescuing the turn to a bigger model, the older
portion of the conversation is summarized (deepseek does the summarizing:
cheap, fast, 1M context), the summary is stored in the session registry, and
the payload is rewritten as [system] + [summary] + [recent turns verbatim].

Why a prefix-hash chain: the client re-sends the full history every turn, so
the router must recognize which leading messages were already summarized. A
rolling hash over messages 0..N does that: if the chain matches the stored
hash at N, history is append-only and the cached summary is still valid. If
it doesn't match (the client edited or reordered history), compaction is
skipped and the overflow rescue handles the turn — never serve a summary that
doesn't match the history it claims to summarize.

Tool-call safety: the compact boundary always sits before a complete
assistant/tool exchange, so no dangling tool_call ids reach the API.
"""
import asyncio
import hashlib
import json

from modules.settings import CONFIG, log, gateway_key

client = None  # shared httpx AsyncClient, set by router.py

COMPACT_KEEP_RECENT = 6      # recent messages always kept verbatim
COMPACT_MIN_SPAN = 4         # don't bother summarizing fewer than this
COMPACT_MAX_OUTPUT = 1200    # summary token budget


def _chain_hashes(messages: list) -> list:
    """Rolling hash chain: hashes[i] covers messages[0..i]."""
    h = ""
    out = []
    for m in messages:
        h = hashlib.sha256((h + json.dumps(m, sort_keys=True)).encode()).hexdigest()
        out.append(h)
    return out


def _compact_boundary(messages: list, span_start: int) -> int:
    """Last index such that messages[boundary:] keeps whole tool exchanges."""
    b = max(span_start, len(messages) - COMPACT_KEEP_RECENT)
    while b > span_start and messages[b].get("role") == "tool":
        b -= 1  # pull the boundary back so tool results stay with their call
    return b


def _summarizer_backend():
    return CONFIG["backends"].get("deepseek")


async def _summarize(client, span: list, prior: str | None) -> str | None:
    b = _summarizer_backend()
    if not b:
        return None
    text = json.dumps(span, indent=1)[:120000]
    prompt = (
        "Summarize this segment of an ongoing agent conversation. The summary "
        "replaces the segment in the conversation history, which continues on a "
        "small-context model. Preserve: decisions made and why, file paths and "
        "key identifiers, tool results later turns depend on, open tasks, and "
        "the user's stated preferences. Omit pleasantries and redundant tool "
        "output. Dense prose or short bullets."
    )
    if prior:
        prompt += f"\n\nAn earlier summary already covers older turns; merge it in:\n{prior}"
    payload = {
        "model": b["model"],
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": text},
        ],
        "max_tokens": COMPACT_MAX_OUTPUT,
    }
    r = await client.post(f"{b['base_url']}/chat/completions", json=payload,
                          headers={"Authorization": f"Bearer {gateway_key()}"})
    if r.status_code != 200:
        log.warning(f"compaction summarizer {r.status_code}: {r.text[:200]}")
        return None
    return r.json()["choices"][0]["message"]["content"]


def _rewrite(messages: list, boundary: int, summary: str) -> list:
    """[system] + [summary] + messages[boundary:]. The summarized span
    (messages up to the boundary) is REPLACED by the summary; the recent
    tail is kept verbatim."""
    header = ("[COMPACTED SESSION SUMMARY — earlier turns were summarized to fit "
              "the current model's context. Treat as verified history.]\n\n")
    summary_msg = {"role": "user", "content": header + summary}
    out = []
    if messages and messages[0].get("role") == "system":
        out.append(messages[0])
        out.append(summary_msg)
        out.extend(messages[boundary:])
    else:
        out.append(summary_msg)
        out.extend(messages[boundary:])
    return out


async def maybe_compact(client, body: dict, sess: dict, sess_key: str) -> dict | None:
    """Attempt compaction for an overflowing k2 turn.

    Returns {"messages": rewritten, "summarized": bool} on success, None when
    compaction isn't possible (too little history, stale prefix, summarizer
    failure) — callers fall back to the overflow rescue.
    """
    messages = body.get("messages", [])
    if len(messages) < COMPACT_KEEP_RECENT + COMPACT_MIN_SPAN + 1:
        return None
    chain = _chain_hashes(messages)

    span_start = 1 if (messages and messages[0].get("role") == "system") else 0
    summary = None
    prior = None
    prev = sess.get("compact")
    if prev:
        n = prev.get("prefix_len", 0)
        if span_start <= n < len(chain) and chain[n - 1] == prev.get("prefix_hash"):
            # stored summary still matches the unchanged history prefix
            span_start = n
            prior = prev.get("summary")

    boundary = _compact_boundary(messages, span_start)
    span = messages[span_start:boundary]
    if len(span) < COMPACT_MIN_SPAN:
        return None

    summary = await _summarize(client, span, prior)
    if not summary:
        return None

    new_prefix_len = boundary
    sess["compact"] = {"prefix_len": new_prefix_len,
                       "prefix_hash": chain[boundary - 1],
                       "summary": summary}
    new_messages = _rewrite(messages, boundary, summary)
    return {"messages": new_messages, "summarized": prior is None,
            "span_messages": len(span)}