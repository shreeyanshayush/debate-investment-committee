"""Risk Agent: holds a single veto. If a stated risk threshold is crossed it forces HOLD over the Judge.

Design choice: the veto decision is made by a deterministic rule engine over the WHOLE fact sheet
(not just the facts the debaters happened to cite), so a risk that neither side emphasised still counts.
A model (in live mode) only writes the narrative; it cannot create, remove or alter a veto.

Veto triggers (any ONE is enough). All limits live in config.py:
  1. A stated metric limit is breached (leverage, customer concentration, cash runway, drawdown).
  2. A hard-flag event is disclosed (going concern, covenant breach, material weakness, ...).
  3. The composite risk score reaches `risk_veto_score`.
"""
from __future__ import annotations

import re
from dataclasses import asdict

from ..config import Config
from ..models import Breach, FactSheet, RiskAssessment
from .base import DebateAgent

_NEGATION = re.compile(r"\b(no|not|without|never|removed|lifted|resolved|cleared)\b[^.;]{0,25}$")

INSTRUCTIONS = """You are the RISK Agent of an investment committee. A deterministic rule engine has already tested the asset
against the committee's stated risk limits; its findings are in `findings`. You hold a veto, but you do not decide it:
the engine does. Write a short plain-English `narrative` (2-4 sentences) explaining the risk picture for the committee,
citing fact IDs from `findings` in [brackets]. Do not add risks that are not in `findings`.
Reply as: {"narrative": "<text>"}"""


def _fmt(v: float, unit: str) -> str:
    return f"{v:g}{unit}"


def evaluate(sheet: FactSheet, cfg: Config) -> RiskAssessment:
    breaches: list[Breach] = []
    flagged_facts: set[str] = set()

    for fid, a in sheet.analysis.items():
        for lim in cfg.limits:
            if lim.key not in a.metrics:
                continue
            v = a.metrics[lim.key]
            crossed = v > lim.limit if lim.kind == "max" else v < lim.limit
            near = (not crossed) and (v >= lim.limit * (1 - cfg.near_limit_band) if lim.kind == "max"
                                      else v <= lim.limit * (1 + cfg.near_limit_band))
            if crossed or near:
                sign = "max" if lim.kind == "max" else "min"
                breaches.append(Breach(rule=lim.key, label=lim.label, kind="limit" if crossed else "near",
                                       observed=_fmt(v, lim.unit), limit=f"{sign} {_fmt(lim.limit, lim.unit)}", cites=[fid]))
                if crossed:
                    flagged_facts.add(fid)

    for fid, fact in sheet.facts.items():
        text = fact.text.lower()
        for flag in cfg.hard_flags:
            for m in re.finditer(re.escape(flag), text):
                if _NEGATION.search(text[max(0, m.start() - 35):m.start()]):
                    continue                                   # e.g. "no going concern doubt"
                breaches.append(Breach(rule="hard_flag", label=f"Hard flag: {flag}", kind="flag",
                                       observed="disclosed", limit="none permitted", cites=[fid]))
                flagged_facts.add(fid)
                break

    n_limit = sum(b.kind == "limit" for b in breaches)
    n_near = sum(b.kind == "near" for b in breaches)
    n_flag = sum(b.kind == "flag" for b in breaches)
    strong_bear = sum(1 for fid, a in sheet.analysis.items()
                      if a.polarity < 0 and a.strength >= 2.0 and fid not in flagged_facts)
    score = min(100, n_limit * cfg.points_breach + n_near * cfg.points_near + n_flag * cfg.points_flag
                + min(cfg.cap_strong_bearish, strong_bear * cfg.points_strong_bearish))

    reasons = [f"{b.label}: {b.observed} vs {b.limit} [{b.cites[0]}]" for b in breaches if b.kind in ("limit", "flag")]
    if score >= cfg.risk_veto_score:
        reasons.append(f"Composite risk score {score} ≥ veto threshold {cfg.risk_veto_score}")
    return RiskAssessment(score=score, breaches=breaches, veto=bool(reasons), trigger_reasons=reasons, narrative="")


def risk_offline(payload: dict) -> dict:
    f = payload["findings"]
    near = [b for b in f["breaches"] if b["kind"] == "near"]
    s = [f"Composite risk score is {f['score']}/100 against a veto threshold of {payload['veto_score']}."]
    if f["veto"]:
        s.append("VETO: " + "; ".join(f["trigger_reasons"]) + ".")
    else:
        s.append("No stated limit or hard flag was crossed.")
    if near:
        s.append("Close to a limit: " + "; ".join(f"{b['label']} {b['observed']} ({b['limit']}) [{b['cites'][0]}]" for b in near) + ".")
    return {"narrative": " ".join(s)}


class RiskAgent(DebateAgent):
    role = "risk"
    retrieval = False
    instructions = INSTRUCTIONS

    async def run(self, sheet: FactSheet) -> RiskAssessment:
        result = evaluate(sheet, self.cfg)
        payload = {"asset": sheet.asset, "veto_score": self.cfg.risk_veto_score, "findings": asdict(result)}
        raw = await self._call(payload, risk_offline)
        narrative = str(raw.get("narrative", "")).strip() or risk_offline(payload)["narrative"]
        result.narrative = narrative                           # narrative only; veto stays rule-based
        return result
