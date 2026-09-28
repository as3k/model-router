"""Persistent usage stats: append-only JSONL (stats.jsonl) + rolling views.

Survives router.log rotation; every route/gate decision appends one line with
append_route(), /v1/recent reads recent_rows(), /v1/stats reads aggregate().
"""
import json

from settings import CONFIG, STATS_FILE, log


def append_route(record: dict):
    try:
        with open(STATS_FILE, "a") as f:
            f.write(json.dumps(record) + "\n")
    except Exception as e:
        log.warning(f"stats append failed: {e}")


def recent_rows(n: int) -> list:
    """Last N decision records, newest first (rich records, unaggregated)."""
    n = max(1, min(n, 100))
    try:
        with open(STATS_FILE) as f:
            rows = [json.loads(l) for l in f if l.strip()]
    except FileNotFoundError:
        rows = []
    return list(reversed(rows[-n:]))


def aggregate() -> dict:
    """Persistent usage stats (survives router.log rotation)."""
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