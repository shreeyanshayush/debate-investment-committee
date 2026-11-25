"""Research Agent: assembles the facts about the asset from the document set (RAG) and classifies them."""
from __future__ import annotations

from ..config import THEMES
from ..heuristics import analyze_fact, extract_metrics
from ..models import AuditEntry, Fact, FactAnalysis, FactSheet
from .base import DebateAgent

# One retrieval query per area an investment committee cares about.
QUERIES = {
    "growth": "revenue growth backlog orders guidance sales",
    "profitability": "operating margin free cash flow earnings profit savings",
    "valuation": "valuation price target analyst rating earnings multiple dividend yield",
    "balance sheet": "net debt leverage liquidity covenant cash runway credit rating",
    "competition": "competitors pricing pressure technology patents market share",
    "regulation & legal": "regulation policy legal investigation audit weakness",
    "customers": "customer concentration largest customer revenue share",
    "management": "chief executive founder insider shares sold bought board",
    "operations": "plant production inventory stores closures",
    "market risk": "share price volatility drawdown stock fell",
}

INSTRUCTIONS = f"""You are the RESEARCH Agent of an investment committee. You are given facts retrieved from the
asset's document set. You may call search_facts to look for anything that is missing.
For EVERY fact in `facts`, classify it:
  theme    : one of {THEMES}
  polarity : +1 if it supports buying the asset, -1 if it argues against, 0 if neutral or mixed
  strength : 0.5 to 3.0 (how decisive the fact is; hard numbers, records and limit breaches score higher)
Do not invent facts and do not change fact IDs or text. Reply as:
{{"analysis": [{{"id": "<fact id>", "theme": "<theme>", "polarity": 1, "strength": 1.5}}]}}"""


class ResearchAgent(DebateAgent):
    role = "research"
    retrieval = True
    instructions = INSTRUCTIONS

    async def run(self) -> FactSheet:
        retrieved: dict[str, list[str]] = {}
        for label, query in QUERIES.items():
            for fact, _score in self.index.search(query, self.cfg.retrieval_top_k):
                retrieved.setdefault(fact.id, []).append(label)
        facts: dict[str, Fact] = {fid: self.index.facts[fid] for fid in sorted(retrieved)}
        cfg = self.cfg

        def offline(payload: dict) -> dict:
            out = []
            for item in payload["facts"]:
                a = analyze_fact(facts[item["id"]], cfg)
                out.append({"id": item["id"], "theme": a.theme, "polarity": a.polarity, "strength": a.strength})
            return {"analysis": out}

        payload = {"asset": self.index.asset_name, "facts": [
            {"id": f.id, "text": f.text, "source": f.source, "section": f.section} for f in facts.values()]}
        raw = await self._call(payload, offline)

        by_id = {str(r.get("id")): r for r in raw.get("analysis", []) if isinstance(r, dict)}
        analysis: dict[str, FactAnalysis] = {}
        for fid, fact in facts.items():
            r = by_id.get(fid)
            fallback = analyze_fact(fact, cfg)
            metrics = extract_metrics(fact.text)          # always deterministic: the veto depends on these
            if r is None:
                self.audit.append(AuditEntry(self.role, "analysis_missing", f"No classification returned for {fid}; rule-based fallback used."))
                analysis[fid] = fallback
                continue
            theme = r.get("theme") if r.get("theme") in THEMES else fallback.theme
            polarity = r.get("polarity") if r.get("polarity") in (-1, 0, 1) else fallback.polarity
            try:
                strength = min(3.0, max(0.5, float(r.get("strength"))))
            except (TypeError, ValueError):
                strength = fallback.strength
            analysis[fid] = FactAnalysis(theme=theme, polarity=int(polarity), strength=round(strength, 2), metrics=metrics)
        return FactSheet(asset=self.index.asset_name, facts=facts, analysis=analysis, retrieved_by=retrieved)
