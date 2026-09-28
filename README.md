# Model Router

A "System 1" decision layer in front of a heterogeneous LLM fleet — one small file that
decides which model every agent request *actually* needs, then dispatches it.

![tier flow](https://img.shields.io/badge/tier-k2%20%E2%86%92%20deepseek%20%E2%86%92%20luna%20%E2%86%92%20terra-blue)

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

## Why it exists

Agentic workloads waste frontier capability on trivial turns, and per-message routing
made multi-turn agent sessions incoherent — models bouncing mid-task, each losing the
thread of its own work. This router treats model selection as a **decision problem**,
solved by purpose-built decision models instead of a bigger LLM guessing.

## Design decisions (each one earned in production)

| Decision | The failure that motivated it |
|---|---|
| **Session pinning** — classify once per session, 0ms after | A 40-turn bug-fix bounced k2→luna→deepseek and flailed for 10 minutes; no model kept its own working memory |
| **Micro-delegation** — short self-contained interjections handled one-off by the local model | Mid-session "what's 45*12?" ran on a frontier model; but session-referential questions ("what was *your* third recommendation?") need context — the criteria defend against that explicitly |
| **Jev is a tiebreaker, not a takeover** — vote clamped to +1 tier | Jev returned p=1.00 on the literal string *"testing testing"* and would route noise to the frontier tier |
| **Escalation-only complexity** | Laya's weakest primitive (ordinal score) inflated trivial input; worst case is now slight overpay, never under-capability |
| **Stakes ≠ complexity** — a board memo is simple to write but high-stakes | Routine/important/critical stakes check gives biz-critical work a frontier brain regardless of complexity |
| **Brain declarations** — crons declare their own tier via `brain: terra` in their system prompt | Only the human knows what matters; classification can't see stakes |
| **Fail-open gate** — watchdog alert triage must never swallow an outage | Deterministic event markers + confidence-earned silence; production: 0 missed events |
| **Overflow rescue + fallback chains** — requests that can't fit the local model are rescued, provider outages degrade gracefully | k2 400s on oversized payloads; a dead provider once stalled a fleet |
| **Monotonic promotion** — sessions promote (k2→deepseek→luna), never demote mid-task | Worst case is slight overpay; never an underpowered answer |

## Production numbers (4 days, 4 machines, real agent fleet)

- **1,600+ routed decisions**, ~210 watchdog gate checks
- **56% of traffic on the free local model**; frontier tier reserved for 15% of requests
- **$2.23 total** estimated spend (deepseek at $0.042/1M input tokens)
- **0 missed watchdog events** across 210 gate checks
- Classification: ~390ms median (local Laya), ~2s when the Jev tiebreaker is consulted
- 6 concurrent classifications complete in 1.8s wall (threaded, non-serializing)

## Endpoints

| Endpoint | Purpose |
|---|---|
| `POST /v1/chat/completions` | OpenAI-compatible proxy (streaming + non-streaming, tools supported) |
| `POST /v1/routing` | Dry-run: tier decision without an LLM call |
| `GET /v1/recent?n=10` | Last N decisions with full decision trail (probability distributions, escalations, text excerpt) |
| `GET /v1/stats` | Aggregates: by tier/via/day, escalations, overflow rescues, cost estimate |
| `GET /v1/gate` | Watchdog alert triage: `notify`/`silent` with fail-open semantics |
| `GET /health` | Liveness + backend list |

Client controls: pin a tier via the `model` field, inline `@tier` tags anywhere in the
last user message (mid-message supported), or declare a session's brain in its system
prompt (`brain: terra`).

## Stack

Python · FastAPI · httpx (async streaming) · [Laya](https://huggingface.co/convaiinnovations/laya)
(Apache 2.0 decision model, CPU-resident) · Jev via Vercel AI Gateway · local
llama-server (llama.cpp fork) · OpenAI Codex (responses-API → chat-SSE conversion) ·
systemd user service on a Tailscale-networked homelab.

## Running it

```bash
pip install fastapi uvicorn httpx "laya[fast]"
uvicorn router:app --host <tailnet-ip> --port 8090
```

Personal paths are env-parameterized (`ROUTER_AUTH_FILE`, `ROUTER_LOG`, `ROUTER_STATS`).
Configure your own backends in `config.json` — the fleet here is an example, not a
requirement.

## Production notes (2.5 days of fleet traffic)

- 1,600+ routing decisions across 4 machines; $2.23 total inference spend
- Watchdog gate: 210 checks, 69 notifies — **all caught by deterministic rules**,
  0 missed events
- Session pinning holds multi-turn agent coherence (see `via=session` records)
- The hardest-won lesson is in the code comments: every guard here was added the
  day something went wrong.
