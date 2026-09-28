"""The questions the decision models are asked. Changing these changes routing
behavior: treat them like config, not comments."""

LAYA_QUESTIONS = {
    "tier": {
        "type": "choice",
        "instructions": "What level of model capability does answering this request well actually require?",
        "criteria": {
            "k2": "short, simple, or mechanical: quick facts, rewrites, small edits, chit-chat, formatting",
            "deepseek": "moderate: multi-step reasoning, code explanation, drafting, summarizing documents",
            "luna": "hard: complex code, architecture, nuanced analysis, needs a frontier model but speed matters",
            "terra": "hardest: deep multi-file reasoning, subtle debugging, high-stakes or very long-chain tasks",
        },
    },
    "complexity": {
        "type": "score",
        "instructions": "How cognitively complex is this request?",
        "criteria": ["trivial", "simple", "moderate", "complex", "very complex"],
    },
}

GATE_QUESTIONS = {
    "notify": {
        "type": "choice",
        "instructions": "This message comes from a production watchdog or monitoring system. Decide whether it should ping the human. The discriminator is EVENTS, not severity: notify whenever anything happened — an action was taken, a container exited or restarted, something was investigated, a problem was found, a threshold was crossed, anything differs from the normal steady state. Only mark silent for pure status messages where nothing happened at all.",
        "criteria": {
            "notify": "an event occurred: container exited/dead/unhealthy or restarted, action taken (restarted, investigated, stopped, started), error or failure, degraded state, warning, threshold crossed, unexpected behavior, something was found or changed — the human should know this happened",
            "silent": "pure routine status with zero events: everything healthy, no actions taken, no events, no thresholds crossed, nothing found, no changes — an unremarkable periodic snapshot or an explicit 'all good, nothing to report'",
        },
    },
}

# MICRO-DELEGATION question: the criteria are written to defend against the
# failure mode observed in production: session-referential interjections
# ("what was your third recommendation?") LOOK self-contained to a classifier
# but require accumulated context. The "session" option explicitly wins ties:
# a missed downshift costs pennies; a context-free answer costs trust.
MICRO_QUESTIONS = {
    "delegate": {
        "type": "choice",
        "instructions": "This is a quick interjection inside an ongoing session that is otherwise doing substantive work. Can this specific message be fully and correctly handled by a small fast local model, without any context from the broader session?",
        "criteria": {
            "k2": "yes: fully self-contained — a quick fact, simple arithmetic, a greeting, a yes/no, a tiny rewrite, a lookup. No reference to anything said or done earlier in the conversation",
            "session": "no: it references the ongoing task or earlier messages (even indirectly — words like 'back to', 'your', 'that', 'it', 'our', 'the recommendation', 'as discussed'), needs the session's accumulated context, involves real reasoning/code/analysis, or is ambiguous. When in doubt, choose session",
        },
    },
}
