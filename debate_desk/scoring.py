"""Deterministic scorecard used by the Judge (and as the offline Judge's whole decision rule).

Only facts that a debater cited can earn credit, and a fact only earns credit for a side if the Research
Agent classified it as supporting that side (or as neutral, at half credit). Citing a bearish fact in a
bull claim earns the bull nothing.
"""
from __future__ import annotations

from .config import Config
from .models import BEAR, BULL, SIGN, Argument, ClaimScore, FactSheet, Rebuttal, ScoreCard


def credit(sheet: FactSheet, fid: str, side: str) -> float:
    a = sheet.analysis[fid]
    w = sheet.weight(fid)
    if a.polarity == SIGN[side]:
        return w
    if a.polarity == 0:
        return 0.5 * w
    return 0.0


def claim_bases(sheet: FactSheet, arg: Argument) -> list[float]:
    """Base weight of each claim; a fact counts once per side even if cited by several claims."""
    seen: set[str] = set()
    out = []
    for c in arg.claims:
        total = 0.0
        for fid in c.cites:
            if fid in seen:
                continue
            seen.add(fid)
            total += credit(sheet, fid, arg.side)
        out.append(total)
    return out


def strongest_claim_index(sheet: FactSheet, arg: Argument) -> int | None:
    bases = claim_bases(sheet, arg)
    return max(range(len(bases)), key=lambda i: bases[i]) if bases else None


def verdict_from_margin(margin: float, cfg: Config) -> str:
    if margin >= cfg.buy_margin:
        return "BUY"
    if margin <= cfg.avoid_margin:
        return "AVOID"
    return "HOLD"


def score_cases(sheet: FactSheet, args: dict[str, Argument], rebuttals: list[Rebuttal], cfg: Config) -> ScoreCard:
    credited: dict[str, dict[str, float]] = {BULL: {}, BEAR: {}}
    base: dict[str, list[float]] = {}
    for side in (BULL, BEAR):
        base[side] = claim_bases(sheet, args[side])
        for c in args[side].claims:
            for fid in c.cites:
                credited[side].setdefault(fid, credit(sheet, fid, side))

    discount = {side: [1.0] * len(args[side].claims) for side in (BULL, BEAR)}
    effects: list[dict] = []
    for r in rebuttals:
        tgt = r.target_side
        valid = (not r.conceded) and r.cites and 0 <= r.target_index < len(base[tgt])
        if not valid:
            effects.append({"rebutting_side": r.side, "target_side": tgt, "target_index": r.target_index,
                            "counter_weight": 0.0, "target_weight": 0.0, "effect": 0.0, "conceded": r.conceded})
            continue
        counter = sum(credit(sheet, f, r.side) for f in dict.fromkeys(r.cites))
        target_w = base[tgt][r.target_index]
        eff = min(1.0, counter / target_w) if target_w > 0 else 0.0
        discount[tgt][r.target_index] *= 1.0 - cfg.rebuttal_max_discount * eff
        effects.append({"rebutting_side": r.side, "target_side": tgt, "target_index": r.target_index,
                        "counter_weight": round(counter, 2), "target_weight": round(target_w, 2),
                        "effect": round(eff, 2), "conceded": False})

    claim_scores: list[ClaimScore] = []
    totals = {}
    for side in (BULL, BEAR):
        total = 0.0
        for i, c in enumerate(args[side].claims):
            eff_w = base[side][i] * discount[side][i]
            claim_scores.append(ClaimScore(side, i, c.text, c.cites, round(base[side][i], 2),
                                           round(discount[side][i], 2), round(eff_w, 2)))
            total += eff_w
        for r in rebuttals:           # facts introduced only in a rebuttal count at reduced credit
            if r.side != side or r.conceded:
                continue
            for f in dict.fromkeys(r.cites):
                if f in credited[side]:
                    continue
                cr = credit(sheet, f, side) * cfg.rebuttal_fact_credit
                credited[side][f] = cr
                total += cr
        totals[side] = total
    denom = totals[BULL] + totals[BEAR]
    margin = (totals[BULL] - totals[BEAR]) / denom if denom > 0 else 0.0
    return ScoreCard(round(totals[BULL], 2), round(totals[BEAR], 2), round(margin, 3), claim_scores,
                     {s: {k: round(v, 2) for k, v in d.items()} for s, d in credited.items()}, effects)
