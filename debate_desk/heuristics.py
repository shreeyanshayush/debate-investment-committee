"""Transparent rule-based fact analysis.

This is the *offline* stand-in for the Research Agent's model call: it assigns each fact a theme, a
polarity (bullish / bearish / neutral) and a strength. It is deliberately simple and inspectable so the
whole system can be demonstrated and tested without an API key. With a live model (Semantic Kernel
backend) the model does the classification instead, and the output is validated against the same schema.

Metric extraction (leverage, customer concentration, runway, drawdown) is ALWAYS done here, by regex,
because the Risk Agent's veto must be auditable and must not depend on a model's judgement.
"""
from __future__ import annotations

import re

from .config import THEMES, Config
from .models import Fact, FactAnalysis

# ---- themes ---------------------------------------------------------------------------------------
# Priority order breaks ties between themes with equal keyword scores.
THEME_ORDER = ["balance_sheet", "customers", "regulation_legal", "governance", "valuation", "market_risk",
               "competition", "operations", "profitability", "growth"]
THEME_KEYWORDS: dict[str, dict[str, int]] = {
    "balance_sheet": {"net debt": 1, "debt": 1, "leverage": 1, "covenant": 1, "runway": 1, "liquidity": 1,
                      "credit rating": 1, "bonds": 1, "cash and": 1, "credit lines": 1, "facilities": 1, "borrow": 1},
    "customers": {"customer": 1},
    "regulation_legal": {"regulat": 1, "policy": 1, "tax-credit": 1, "legislat": 1, "investigation": 1,
                         "lawsuit": 1, "enforcement": 1, "material weakness": 1, "litigation": 1},
    "governance": {"chief executive": 1, "founder": 1, "insider": 1, "board": 1, "director": 1},
    "valuation": {"forward earnings": 1, "forward revenue": 1, "price target": 1, "discount to": 1, "dividend": 1,
                  "yield": 1, "analyst": 1, "rate the stock": 1, "downgraded": 1, "consensus": 1, "payout": 1,
                  "brokers": 1, "estimates": 1},
    "market_risk": {"stock fell": 1, "drawdown": 1, "volatility": 1},
    "competition": {"competitor": 1, "rival": 1, "pricing pressure": 1, "cut prices": 1, "selling prices": 1,
                    "patent": 1, "market share": 1, "efficiency": 1, "technology": 1},
    "operations": {"plant": 1, "fire": 1, "stores": 1, "inventory": 1, "closure": 1, "halted": 1},
    "profitability": {"margin": 2, "cash flow": 1, "earnings": 1, "profit": 1, "ebitda": 1, "savings": 1,
                      "write-down": 1, "charges": 1},
    "growth": {"revenue": 1, "growth": 1, "backlog": 1, "order": 1, "guidance": 1, "sales": 1, "sessions": 1,
               "grew": 1, "rose": 1},
}

# ---- polarity lexicon (regex -> weight). Longer phrases are matched first and masked. ---------------
POSITIVE = {
    r"turned positive": 1.5, r"costs? fell": 1.0, r"no covenant pressure": 1.0, r"has not sold any": 1.0,
    r"insurance is expected to cover most of the loss": 1.5, r"ahead of": 1.0, r"track record": 1.0,
    r"discount to": 1.0, r"\bwell below\b|below the [\w\s-]{0,25}(?:median|average)": 1.0, r"\brecord\b": 1.0, r"\btripled\b": 1.5, r"\brose\b": 1.0, r"\bgrew\b": 1.0,
    r"\bgrowth\b": 0.5, r"\bimproved\b": 1.0, r"\bexpanded\b": 1.0, r"\braised\b": 1.0, r"\bupside\b": 1.0,
    r"\bpatent\w*": 1.0, r"\blifted\b": 1.0, r"\bsigned\b": 1.0, r"\bresumed\b": 1.0, r"\bdelivered\b": 1.0,
    r"\bpurchased\b": 1.0, r"\badded\b": 0.5, r"\brate the stock buy\b": 1.0, r"\bundrawn\b": 0.5,
    r"under a pre-arranged trading plan": 0.5, r"\bcuts charging time\b": 1.5,
}
NEGATIVE = {
    r"compared with an outflow of": 0.0, r"taking market share": 1.5, r"exceeds 100%": 1.0,
    r"depends on": 1.0, r"cut estimates": 1.0, r"cut prices": 1.0, r"write-down": 1.5, r"\bdowngraded\b": 1.5,
    r"\bfell\b": 1.0, r"\bdeclines?\b": 1.0, r"\bdeclined\b": 1.0, r"\bbreach\w*": 2.0, r"\bwaiver\b": 1.0,
    r"\bweakness\b": 1.5, r"\buncertain\w*": 1.0, r"\bpressure\b": 1.0, r"\bcompress\w*": 1.0,
    r"\bclose\b": 1.0, r"\boutflow\b": 1.0, r"\bburn\b": 0.5, r"\bsold\b": 1.0, r"\bhalted\b": 1.0,
    r"\bfire\b": 1.0, r"\bexpires?\b": 1.0, r"\bweaker\b": 1.0, r"\bweak\b": 1.0, r"\bnarrowed\b": 1.0,
    r"\bshrink\b": 1.0, r"\bmarkdowns\b": 1.0, r"\brestructuring\b": 1.0, r"\btrimming\b": 1.0,
    r"\bconcern\w*": 1.0, r"\bunderperforming\b": 1.0, r"\bloss\b": 1.0, r"\blawsuit\b": 1.5,
    r"\binvestigation\b": 1.5, r"\bconcentration\b": 1.0,
}
NEGATORS = {"no", "not", "without", "never", "hasn't", "has not"}
EMPHASIS = [r"\brecord\b", r"\btripled\b", r"consecutive", r"second time", r"\bmaterial\b", r"turned positive",
            r"\bbreach", r"downgrad", r"taking market share", r"exceeds 100%"]
NEUTRAL_BAND = 0.75


def _score(text: str) -> float:
    t = text.lower()
    total = 0.0
    entries = [(p, w, 1) for p, w in POSITIVE.items()] + [(p, w, -1) for p, w in NEGATIVE.items()]
    entries.sort(key=lambda e: -len(e[0]))          # longer patterns first, then mask what they matched
    for pattern, weight, sign in entries:
        for m in list(re.finditer(pattern, t)):
            before = t[max(0, m.start() - 18):m.start()].split()[-2:]
            negated = any(w in NEGATORS for w in before)
            total += weight * sign * (-1 if negated else 1)
            t = t[:m.start()] + " " * (m.end() - m.start()) + t[m.end():]
    return total


def _theme(text: str) -> str:
    t = text.lower()
    best, best_score = "growth", -1
    for theme in THEME_ORDER:
        s = sum(w for kw, w in THEME_KEYWORDS[theme].items() if kw in t)
        if s > best_score:
            best, best_score = theme, s
    return best


# ---- metric extraction (deterministic, used by the Risk Agent) -------------------------------------
_LEV = re.compile(r"net debt[- ]to[- ]ebitda[^.\d]{0,25}(\d+(?:\.\d+)?)\s*(?:x|times)")
_CONC = re.compile(r"(?:largest|top[\w\s-]{0,14}?) customers?[^.%]{0,70}?(\d+(?:\.\d+)?)\s*%")
_RUNWAY = re.compile(r"(\d+(?:\.\d+)?)\s*months?\s+of\s+(?:cash\s+)?runway")
_DRAW = re.compile(r"drawdown[^.\d]{0,40}(\d+(?:\.\d+)?)\s*%|as\s+much\s+as\s+(\d+(?:\.\d+)?)\s*%\s+(?:peak|from)")


def extract_metrics(text: str) -> dict[str, float]:
    t = text.lower()
    out: dict[str, float] = {}
    if m := _LEV.search(t):
        out["leverage_x"] = float(m.group(1))
    if m := _CONC.search(t):
        out["customer_concentration_pct"] = float(m.group(1))
    if m := _RUNWAY.search(t):
        out["runway_months"] = float(m.group(1))
    if m := _DRAW.search(t):
        out["drawdown_pct"] = float(m.group(1) or m.group(2))
    return out


def _metric_polarity(metrics: dict[str, float], cfg: Config) -> tuple[int, float] | None:
    lim = {l.key: l.limit for l in cfg.limits}
    if "leverage_x" in metrics:
        v = metrics["leverage_x"]
        return (-1, 3.0) if v >= lim["leverage_x"] else (+1, 1.5) if v <= 2.0 else (-1, 1.5)
    if "customer_concentration_pct" in metrics:
        v = metrics["customer_concentration_pct"]
        return (-1, 3.0) if v >= lim["customer_concentration_pct"] else (-1, 1.5) if v >= 30 else (+1, 1.0)
    if "runway_months" in metrics:
        v = metrics["runway_months"]
        return (-1, 3.0) if v < lim["runway_months"] else (+1, 1.5) if v >= 24 else (0, 1.0)
    if "drawdown_pct" in metrics:
        v = metrics["drawdown_pct"]
        return (-1, 2.5) if v >= lim["drawdown_pct"] else (-1, 1.0) if v >= 20 else (0, 1.0)
    return None


def analyze_fact(fact: Fact, cfg: Config | None = None) -> FactAnalysis:
    # Classification uses FIXED reference thresholds, deliberately NOT the caller's risk limits: the Risk
    # Agent's limits must not change how the Research/Judge side reads a fact, or the two would not be
    # independent and a veto could be "pre-baked" into the Judge's verdict.
    metrics = extract_metrics(fact.text)
    theme = _theme(fact.text)
    if (mp := _metric_polarity(metrics, Config())) is not None:
        polarity, strength = mp
    else:
        s = _score(fact.text)
        polarity = 1 if s >= NEUTRAL_BAND else -1 if s <= -NEUTRAL_BAND else 0
        has_number = bool(re.search(r"\d", fact.text))
        emph = sum(1 for p in EMPHASIS if re.search(p, fact.text.lower()))
        strength = min(3.0, 1.0 + 0.5 * has_number + 0.5 * min(2, emph))
    assert theme in THEMES
    return FactAnalysis(theme=theme, polarity=polarity, strength=round(strength, 2), metrics=metrics)
