"""Rebuttal round: each side responds ONCE to the other side's strongest point, with cited evidence.

The opponent's strongest point is chosen by code (highest evidence weight), not by the debater, so a
side cannot dodge the hard question. Counter-evidence is retrieved from the document set by similarity
to the point being answered. If no supporting evidence exists, the side must concede the point.
"""
from __future__ import annotations

import asyncio

from ..citations import clean_cites, numbers_grounded
from ..models import BEAR, BULL, OPPONENT, SIGN, Argument, AuditEntry, FactSheet, Rebuttal
from ..scoring import strongest_claim_index
from .base import DebateAgent, THEME_LABEL

MIN_SIMILARITY = 0.05
MAX_COUNTERS = 2

_RULES = """Rules:
- Answer ONLY the single point in `target`. Respond once.
- Support your reply with facts from `candidate_evidence` (or facts you find with search_facts) and cite their IDs.
- Never state a number that is not in a fact you cite or in the target's cited facts.
- If no fact genuinely answers the point, concede it: set "conceded": true and "cites": [].
Reply as: {"response": "<2-3 sentences>", "cites": ["<fact id>"], "conceded": false}"""

INSTRUCTIONS = {
    BULL: "You are the BULL Agent in the rebuttal round. Rebut the Bear's strongest point.\n" + _RULES,
    BEAR: "You are the BEAR Agent in the rebuttal round. Rebut the Bull's strongest point.\n" + _RULES,
}


def rebuttal_offline(payload: dict) -> dict:
    cands = payload["candidate_evidence"][:MAX_COUNTERS]
    if not cands:
        return {"response": "Conceded: the record contains no evidence that answers this point.", "cites": [], "conceded": True}
    theme = THEME_LABEL.get(payload["target"]["theme"], payload["target"]["theme"])
    parts = " ".join(f"{c['text']} [{c['id']}]" for c in cands)
    return {"response": f"On the {payload['opponent']}'s {theme.lower()} point: {parts}",
            "cites": [c["id"] for c in cands], "conceded": False}


class RebuttalRound(DebateAgent):
    role = "rebuttal"
    retrieval = True

    async def run(self, sheet: FactSheet, args: dict[str, Argument]) -> list[Rebuttal]:
        return list(await asyncio.gather(self._respond(BULL, sheet, args), self._respond(BEAR, sheet, args)))

    async def _respond(self, side: str, sheet: FactSheet, args: dict[str, Argument]) -> Rebuttal:
        opp = OPPONENT[side]
        tidx = strongest_claim_index(sheet, args[opp])
        if tidx is None:
            return Rebuttal(side, opp, -1, f"The {opp} presented no cited case, so there is nothing to rebut.", [], True)
        target = args[opp].claims[tidx]
        target_themes = [sheet.analysis[f].theme for f in target.cites]
        own_cited = {f for c in args[side].claims for f in c.cites}
        sims = {f.id: s for f, s in self.index.search(target.text, k=self.cfg.retrieval_top_k * 2)}

        cands = []
        for fid in sheet.facts:
            a = sheet.analysis[fid]
            if a.polarity != SIGN[side]:
                continue
            sim, same_theme = sims.get(fid, 0.0), a.theme in target_themes
            if not (same_theme or sim >= MIN_SIMILARITY):
                continue
            rank = sim + (0.5 if same_theme else 0.0) + 0.05 * sheet.weight(fid) + (0.2 if fid not in own_cited else 0.0)
            cands.append((rank, fid))
        cands.sort(reverse=True)
        candidate_evidence = [{"id": fid, "text": sheet.facts[fid].text, "theme": sheet.analysis[fid].theme,
                               "weight": round(sheet.weight(fid), 2), "already_cited_by_you": fid in own_cited}
                              for _, fid in cands[:4]]
        payload = {"side": side, "opponent": opp, "asset": sheet.asset,
                   "target": {"index": tidx, "text": target.text, "cites": target.cites,
                              "theme": target_themes[0] if target_themes else "growth",
                              "cited_facts": [{"id": f, "text": sheet.facts[f].text} for f in target.cites]},
                   "candidate_evidence": candidate_evidence}
        raw = await self.backend.run(f"rebuttal_{side}", INSTRUCTIONS[side], payload, rebuttal_offline, retrieval=True)

        agent = f"rebuttal_{side}"
        self.admit(sheet, [str(c).strip("[]") for c in raw.get("cites", []) or []])
        cites, rejected = clean_cites(raw.get("cites"), set(self.index.facts))
        if rejected:
            self.audit.append(AuditEntry(agent, "invalid_citation", f"Stripped unknown IDs {rejected} from the rebuttal."))
        response = str(raw.get("response", "")).strip()
        pool = [sheet.facts[c].text for c in cites] + [sheet.facts[f].text for f in target.cites]
        bad = numbers_grounded(response, pool)
        if bad:
            self.audit.append(AuditEntry(agent, "ungrounded_number", f"Rebuttal dropped; numbers {bad} not in cited facts."))
            cites, response = [], ""
        conceded = bool(raw.get("conceded")) or not cites
        if conceded and not cites and not response:
            response = "Conceded: the record contains no evidence that answers this point."
        return Rebuttal(side=side, target_side=opp, target_index=tidx, response=response, cites=cites, conceded=conceded)
