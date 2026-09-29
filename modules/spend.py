"""Daily spend guardrails: in-memory per-day cost tracking seeded from
stats.jsonl, soft/hard limits, and the degraded-mode flag escalations honor.

Accounting is estimate-based, mirroring /v1/stats: seed = (sum over today's
route records of est_input_tokens per tier * stats_prices_per_mtok_input),
then each request adds its estimated cost as it dispatches. The counter rolls
over at local midnight (checked on every note_usage call). Soft crossing fires
a /v1/gate POST (via router.py, fire-and-forget); hard crossing switches the
router to degraded mode until midnight: jev tiebreaker is skipped and k2
overflow rescue stays on k2-with-compaction instead of deepseek. Explicit
@tier/forced requests keep working: degraded mode never touches forced paths.
"""
import json
import time

from modules.settings import CONFIG, STATS_FILE, log

client = None  # shared httpx AsyncClient, set by router.py


class DailySpend:
    def __init__(self, seed: bool = True):
        self._day = time.strftime("%Y-%m-%d")
        self.total = 0.0
        self.soft_fired = False   # gate POST already fired for this soft crossing
        self.hard_fired = False   # degraded-mode entry already logged this day
        self.hard_active = False  # degraded mode until the next midnight roll
        if seed:
            self._seed()

    # ---- config ----
    @staticmethod
    def _prices() -> dict:
        return CONFIG.get("stats_prices_per_mtok_input", {})

    @staticmethod
    def soft_limit() -> float:
        return CONFIG.get("spend_guardrails", {}).get("daily_soft_usd", 2.0)

    @staticmethod
    def hard_limit() -> float:
        return CONFIG.get("spend_guardrails", {}).get("daily_hard_usd", 5.0)

    # ---- startup seed: replay today's routes from stats.jsonl ----
    def _seed(self):
        prices = self._prices()
        tot = 0.0
        try:
            with open(STATS_FILE) as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        r = json.loads(line)
                    except Exception:
                        continue
                    if r.get("type", "route") != "route":
                        continue
                    if (r.get("ts") or "")[:10] != self._day:
                        continue
                    tot += (r.get("est_tokens") or 0) / 1_000_000 * prices.get(r.get("tier"), 0)
        except FileNotFoundError:
            pass
        self.total = tot
        if tot >= self.hard_limit():
            self.hard_active = True
            self.hard_fired = True
            log.warning(f"spend guard: seeded at ${tot:.2f}, past hard limit, degraded mode active")
        elif tot >= self.soft_limit():
            self.soft_fired = True
            log.warning(f"spend guard: seeded at ${tot:.2f}, already past soft limit today")

    # ---- per-request accounting ----
    def _roll(self):
        today = time.strftime("%Y-%m-%d")
        if today != self._day:  # midnight: fresh day, fresh counters
            self._day = today
            self.total = 0.0
            self.soft_fired = False
            self.hard_fired = False
            self.hard_active = False

    @staticmethod
    def cost_of(tier: str, est_tokens: int) -> float:
        return est_tokens / 1_000_000 * DailySpend._prices().get(tier, 0)

    def note_usage(self, tier: str, est_tokens: int) -> dict:
        """Add one request's estimated cost. Returns {"soft": bool, "hard": bool}
        crossed-this-request events; each limit fires once per day."""
        self._roll()
        self.total += self.cost_of(tier, est_tokens)
        events = {"soft": False, "hard": False}
        if not self.soft_fired and self.total >= self.soft_limit():
            self.soft_fired = True
            events["soft"] = True
        if not self.hard_fired and self.total >= self.hard_limit():
            self.hard_fired = True
            self.hard_active = True
            events["hard"] = True
        return events


tracker = DailySpend()