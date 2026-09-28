# Model Router

A small FastAPI service that sits between my agent fleet and its models. Every
request from every agent on my network passes through it, gets classified by a
tiny decision model, and gets sent to the cheapest model that can actually do
the job. In practice that's a cheap cloud model (deepseek at $0.042/1M tokens)
for most full agentic sessions, a free 4B model on my GPU for short contexts
and quick interjections, and GPT-5.6 only for the tasks that earn it.

```
incoming request ──► LAYA (421M decision model, local, ~35ms)
                      │  "what capability does this need?"  (choice)
                      │  "how complex is it?"               (score)
                      │  "what happens if this is wrong?"   (stakes)
                      ▼ unsure? ──► JEV (commercial tiebreaker, clamped ±1 tier)
                      ▼
      k2 (local 4B) ──► deepseek-v4-flash ──► gpt-5.6-luna ──► gpt-5.6-terra
      free/private        ~$0.042/1M            frontier (Codex OAuth)
```

## Why I built it

Two problems, and they fed each other.

First, model selection was solved badly. Each message got classified on its own,
so a 40-turn bug fix bounced between three models and none of them kept the
thread of their own work. I watched a cron fix take 10 minutes on a task that
needed tools added, and the reason was right there in the logs: the session
couldn't hold one brain.

Second, the fleet paid frontier prices for turns like "what's 2+2".

So model selection became a decision problem, and I gave it to decision models
instead of a bigger LLM guessing. Laya (a 421M classifier on my CPU) answers
three questions per request: what capability does this need, how complex is it,
and what happens if it's wrong. If Laya is unsure, Jev gets a vote. Then the
request goes to the cheapest tier that can handle it.

## The parts that came from real failures

| Decision | The failure behind it |
|---|---|
| Session pinning: classify once, then 0ms | A 40-turn bug fix bounced k2→luna→deepseek and flailed for 10 minutes. No model kept its own working memory |
| Micro-delegation for quick interjections | Mid-session "what's 45*12?" ran on a frontier model. But "what was your third recommendation?" needs session context, so the classifier criteria reject anything that references the ongoing work |
| Jev can only tiebreak, never take over | Jev returned p=1.00 on the literal string "testing testing" and would route noise to the frontier tier. Its vote is now clamped to one tier above Laya's |
| Complexity escalates, never demotes | Laya's score primitive inflated trivial input and over-escalated to paid models. Worst case now is slight overpay, never an underpowered answer |
| Stakes are not complexity | A board memo is simple to write but high stakes. A routine/important/critical check gives biz-critical work the frontier brain regardless of complexity |
| Crons declare their own brain | A `brain: terra` line in the system prompt beats classification. Only the human knows what matters |
| Fail-open watchdog gate | An alert gate that swallows an outage is worse than a redundant ping. Production: no missed events observed |
| Overflow rescue and fallback chains | Requests that can't fit the local model get rescued instead of 400ing. A dead provider degrades instead of stalling the fleet |

## Numbers from the first 4 days

1,606 routing decisions and 216 watchdog gate checks across 3 machines
(nexus, lunamor, oathgate).

- deepseek handled 72% of routes, luna 15%, k2 12%. Terra ran 26 times.
- Total inference spend: $2.25 (deepseek at $0.042/1M input tokens, everything
  else free or subscription)
- Watchdog gate: 216 checks, 71 notifies. Every notify came from
  deterministic event rules, and every silence was reviewed against the source
  alert. No swallowed alerts found.
- Classification overhead: ~1.5s median when laya decides (three questions on
  CPU), ~2s when jev gets consulted, 0 on pinned turns.
- 6 parallel classifications finish in 1.9s wall; per-probe latency barely moves
  with concurrency (threaded, not serialized).

## Endpoints

| Endpoint | Purpose |
|---|---|
| `POST /v1/chat/completions` | OpenAI-compatible proxy (streaming and non-streaming, tools supported) |
| `POST /v1/routing` | Dry run: tier decision without an LLM call |
| `GET /v1/recent?n=10` | Last N decisions with the full trail: probability distributions, escalations, text excerpt |
| `GET /v1/stats` | Aggregates by tier/via/day, escalations, overflow rescues, cost estimate |
| `GET /v1/gate` | Watchdog alert triage: notify or silent, fail-open |
| `GET /health` | Liveness + backend list |

Clients can pin a tier with the `model` field, drop an `@tier` tag anywhere in
the last user message, or declare a session brain in the system prompt
(`brain: terra`).

## Stack

Python, FastAPI, httpx for async streaming, [Laya](https://huggingface.co/convaiinnovations/laya)
(421M decision model, Apache 2.0, runs on CPU), Jev through Vercel AI Gateway, a
local llama.cpp server for the 4B tier, and OpenAI Codex behind a responses-API
to chat-SSE converter. Runs as a systemd user service on a Tailscale-networked
homelab.

## Running it

```bash
pip install fastapi uvicorn httpx "laya[fast]"
uvicorn router:app --host <tailnet-ip> --port 8090
```

Personal paths are env-parameterized (`ROUTER_AUTH_FILE`, `ROUTER_LOG`,
`ROUTER_STATS`). Set up your own backends in `config.json`; the fleet here is
an example, not a requirement.

## What the first 4 days taught me

The guards in this code are all scar tissue. Context overflow taught the
rescue logic. Jev's confidence on the string "testing testing" taught the
clamp. Session-referential interjections that fooled the micro-delegation
check taught the stricter criteria. A swallowed watchdog alert would have been
worse than all of them, which is why the gate fails open. Every guard here
went in the day something went wrong.
