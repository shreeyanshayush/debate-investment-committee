"""Shared agent plumbing: context, base classes, and the offline opening-argument logic."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from ..citations import validate_claims
from ..config import Config
from ..heuristics import analyze_fact
from ..llm import Backend
from ..models import BULL, SIGN, Argument, AuditEntry, FactSheet
from ..rag import DocumentIndex

THEME_LABEL = {
    "growth": "Growth", "profitability": "Profitability", "balance_sheet": "Balance sheet",
    "regulation_legal": "Regulation & legal", "valuation": "Valuation", "competition": "Competition",
    "customers": "Customers", "governance": "Management & governance", "market_risk": "Market risk",
    "operations": "Operations",
}


@dataclass
class Context:
    index: DocumentIndex
    cfg: Config
    audit: list[AuditEntry] = field(default_factory=list)


class DebateAgent:
    role = "agent"
    instructions = ""
    retrieval = False        # whether the agent is given the RAG search tool in live mode

    def __init__(self, backend: Backend, ctx: Context):
        self.backend, self.ctx = backend, ctx
        self.cfg, self.index, self.audit = ctx.cfg, ctx.index, ctx.audit

    async def _call(self, payload: dict[str, Any], offline: Callable[[dict], dict]) -> dict:
        return await self.backend.run(self.role, self.instructions, payload, offline, retrieval=self.retrieval)

    # ---- helpers shared by the debating agents ----
    def admit(self, sheet: FactSheet, cites: list[str]) -> None:
        """A cited fact that exists in the index but was not in the fact sheet (found via live RAG) is
        admitted, classified by the rule-based analyser, and logged."""
        for fid in cites:
            if fid not in sheet.facts and fid in self.index.facts:
                fact = self.index.facts[fid]
                sheet.facts[fid] = fact
                sheet.analysis[fid] = analyze_fact(fact, self.cfg)
                sheet.retrieved_by.setdefault(fid, []).append("late-retrieval")
                self.audit.append(AuditEntry(self.role, "fact_admitted", f"{fid} was not in the Research fact sheet; admitted via RAG."))

    def fact_text(self) -> dict[str, str]:
        return {fid: f.text for fid, f in self.index.facts.items()}


def fact_view(sheet: FactSheet, fid: str) -> dict[str, Any]:
    f, a = sheet.facts[fid], sheet.analysis[fid]
    return {"id": fid, "text": f.text, "source": f.source, "section": f.section, "theme": a.theme,
            "polarity": a.polarity, "strength": a.strength, "weight": round(sheet.weight(fid), 2)}


def opening_offline(payload: dict) -> dict:
    """Pick the side's strongest facts, one claim per theme first (diversity), then fill by weight."""
    side, limit = payload["side"], payload["max_claims"]
    sign = SIGN[side]
    facts = sorted((f for f in payload["fact_sheet"] if f["polarity"] == sign), key=lambda f: -f["weight"])
    chosen, themes = [], set()
    for f in facts:
        if f["theme"] not in themes and len(chosen) < limit:
            chosen.append(f)
            themes.add(f["theme"])
    for f in facts:
        if f not in chosen and len(chosen) < limit:
            chosen.append(f)
    chosen.sort(key=lambda f: -f["weight"])
    return {"claims": [{"text": f"{THEME_LABEL.get(f['theme'], f['theme'])}: {f['text']}", "cites": [f["id"]]} for f in chosen]}


class DebaterAgent(DebateAgent):
    side = BULL
    retrieval = True

    async def run(self, sheet: FactSheet) -> Argument:
        payload = {"side": self.side, "asset": sheet.asset, "max_claims": self.cfg.max_claims,
                   "fact_sheet": [fact_view(sheet, fid) for fid in sheet.facts]}
        raw = await self._call(payload, opening_offline)
        raw_claims = raw.get("claims", [])
        for rc in raw_claims:                      # admit RAG-found facts before validating
            self.admit(sheet, [str(c).strip("[]") for c in rc.get("cites", []) or []])
        claims = validate_claims(self.role, raw_claims, self.fact_text(), self.audit)[: self.cfg.max_claims]
        if not claims:
            self.audit.append(AuditEntry(self.role, "no_valid_case", f"{self.role} produced no validly cited claim; its case is empty."))
        return Argument(side=self.side, claims=claims)
