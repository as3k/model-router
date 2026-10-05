"""Persistent usage stats: append-only JSONL (stats.jsonl) + rolling views.

Survives router.log rotation; every route/gate decision appends one line with
append_route(), /v1/recent reads recent_rows(), /v1/stats reads aggregate().
"""
import json

from modules.settings import CONFIG, STATS_FILE, log


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

# ---- outcome feedback loop (backlog 7) -------------------------------------
# Signals, in order of trust:
#   1. explicit: agents/crons POST /v1/feedback with a verdict on a turn
#   2. derived:  the user re-asking the same thing shortly after a turn is the
#      classic "that answer failed" signal, detectable from the logged excerpts
# Both land in stats.jsonl so history accumulates; /v1/outcomes aggregates.

def append_outcome(record: dict):
    record["type"] = "outcome"
    try:
        with open(STATS_FILE, "a") as f:
            f.write(json.dumps(record) + "\n")
    except Exception as e:
        log.warning(f"outcome append failed: {e}")


def _norm_text(text: str) -> str:
    import re
    t = (text or "").lower()
    t = re.sub(r"^user:\s*", "", t)
    return re.sub(r"\s+", " ", t).strip()


def find_reasks(rows: list, window_min: int = 30, ratio: float = 0.8) -> list:
    """Detect likely retries: a user ask that closely repeats an ask from the
    same or another session within the window. The EARLIER turn is the one
    that presumably failed (the user came back with the same question)."""
    import difflib
    import time as _time
    routes = [r for r in rows if r.get("type", "route") == "route"]
    routes.sort(key=lambda r: r.get("ts", ""))
    texts = []
    for r in routes:
        # compare the USER ASK, not the full state: full-state excerpts of
        # consecutive turns in one session are similar by construction (tool
        # output context) and would flag ordinary task progress as re-asks.
        t_raw = r.get("user_text")
        if not t_raw:
            ex = r.get("state_excerpt") or ""
            # legacy records: best-effort extraction of the trailing user turn
            if "user:" in ex and not ex.strip().startswith("tool:"):
                t_raw = ex[ex.rfind("user:") + 5:]
        if not t_raw or len(t_raw.strip()) < 12:
            continue
        texts.append((r, _norm_text(t_raw)))
    out = []
    for i, (r_later, t_later) in enumerate(texts):
        if len(t_later) < 12:  # too short to judge ("ok", "yes")
            continue
        for j in range(i - 1, -1, -1):
            r_earlier, t_earlier = texts[j]
            if len(t_earlier) < 12:
                continue
            gap_min = _gap_minutes(r_earlier.get("ts"), r_later.get("ts"))
            if gap_min is None or gap_min > window_min:
                break  # sorted by time; everything older is outside the window
            if gap_min < 1 and r_earlier.get("session") == r_later.get("session"):
                continue  # same turn re-classified, not a re-ask
            sim = difflib.SequenceMatcher(None, t_earlier, t_later).ratio()
            if sim >= ratio:
                out.append({"reask_ts": r_later.get("ts"), "original_ts": r_earlier.get("ts"),
                            "tier_original": r_earlier.get("tier"), "via_original": r_earlier.get("via"),
                            "session_original": (r_earlier.get("session") or "")[:8],
                            "session_reask": (r_later.get("session") or "")[:8],
                            "similarity": round(sim, 2), "gap_min": round(gap_min, 1),
                            "excerpt": t_earlier[:120]})
                break  # one re-ask pairing per later turn
    return out


def _gap_minutes(ts_early: str, ts_late: str) -> float | None:
    from datetime import datetime
    try:
        a = datetime.strptime(ts_early[:19], "%Y-%m-%dT%H:%M:%S")
        b = datetime.strptime(ts_late[:19], "%Y-%m-%dT%H:%M:%S")
        return (b - a).total_seconds() / 60
    except Exception:
        return None


def outcomes_summary() -> dict:
    """Tier-level outcome view: which tier's turns get re-asked, escalated,
    or overflow-rescued. This is the evidence base for tuning thresholds."""
    import json as _json
    rows = []
    try:
        with open(STATS_FILE) as f:
            for line in f:
                try:
                    rows.append(_json.loads(line))
                except Exception:
                    pass
    except FileNotFoundError:
        pass
    routes = [r for r in rows if r.get("type", "route") == "route"]
    outcomes = [r for r in rows if r.get("type") == "outcome"]
    reasks = find_reasks(rows)
    reask_by_tier = {}
    for ra in reasks:
        t = ra.get("tier_original", "?")
        reask_by_tier[t] = reask_by_tier.get(t, 0) + 1
    turns_by_tier = {}
    for r in routes:
        t = r.get("tier", "?")
        turns_by_tier[t] = turns_by_tier.get(t, 0) + 1
    per_tier = {}
    for t, total in turns_by_tier.items():
        rk = reask_by_tier.get(t, 0)
        per_tier[t] = {"turns": total, "reasks": rk,
                       "reask_rate": round(rk / total, 3) if total else 0.0}
    qe = [r for r in routes if r.get("escalated_by") == "quality" or
          "quality" in str(r.get("escalated_by", ""))]
    return {
        "routes_analyzed": len(routes),
        "explicit_feedback": len(outcomes),
        "reasks_detected": len(reasks),
        "reask_pairs": reasks[-10:],
        "per_tier": per_tier,
        "quality_escalations": len(qe),
        "overflow_rescues": sum(1 for r in routes if r.get("overflow")),
        "compactions": sum(1 for r in routes if r.get("compacted")),
    }
