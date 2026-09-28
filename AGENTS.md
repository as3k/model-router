# Router — agent notes

Read this before touching anything in this directory. This is a live service:
`uvicorn router:app` (systemd user unit `router.service`) serving the whole
agent fleet over Tailscale. If routing breaks, every agent feels it.

## What lives here

| File | Role |
|---|---|
| `router.py` | FastAPI app + endpoints + session orchestration (entry point) |
| `modules/classify.py` | Laya/Jev deciders, routing policy, gate logic |
| `modules/questions.py` | Decision-model question dicts (these steer behavior) |
| `modules/sessions.py` | Session registry: pin, promote, expire |
| `modules/dispatch.py` | Backend dispatch, streaming, fallback chains |
| `modules/stats_store.py` | stats.jsonl persistence + aggregation |
| `modules/settings.py` | Config, paths, credentials |
| `config.json` | Live config (tiers, thresholds, brain rules, prices) |
| `stats.jsonl` | Append-only decision trail (survives log clears) |
| `sessions.json` | Active pinned sessions |
| `router.log` | Human-readable decision log (clear anytime) |

Full docs: `~/Mycelium/Atlas/Systems/model-router/` (index = architecture,
api = calling guide, agent-guide = etiquette, hermes/remote-pi-setups = fleet).

## Rules for editing

1. **Backup before you edit, then clean up after yourself.** Copy it to `backups/<name>.pre-<what>-<date>` (create the folder if needed), make your change,
   verify the service restarts and passes a smoke check, then delete the backup.
   Do not leave `.pre-*` files in the router root.
2. **Restart after config/code changes:** `systemctl --user restart router`,
   then check `/health` and one `/v1/routing` dry-run before walking away.
3. **Never change the question strings in `questions.py` casually.** They steer
   the classifier. Em dashes and phrasing in them are functional. Test after.
4. **`stats.jsonl` is append-only history** — do not truncate it for cleanup
   unless Z asks. `router.log` can be cleared freely.
5. **Concurrent edits happen.** More than one agent session works on this
   router. Before editing, check the file mtime; after finishing, commit and
   push so the other session rebases instead of clobbering.
6. **Test battery after every change** (all must pass):
   - `/health` returns ok with all backends
   - `/v1/routing` dry-run returns a tier
   - `/v1/gate` with "container exited" returns notify via rule
   - `/v1/gate` with "All systems healthy" returns silent
   - Two-turn session pins (`via=session` on turn 2)
   - `/v1/stats` returns valid JSON
7. **Never expose the router beyond the tailnet.** No auth on this service;
   tailnet membership is the trust boundary.
8. **Never route gpt models through Vercel AI Gateway.** Luna/terra are Codex
   OAuth, read from the pi auth file. Layas run on CPU only — a second GPU
   laya instance next to k2 has OOM-killed the local model twice.

## Verify before you claim done

```bash
systemctl --user is-active router
curl -s http://100.118.202.118:8090/health
curl -s http://100.118.202.118:8090/v1/routing -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"test"}]}'
```
