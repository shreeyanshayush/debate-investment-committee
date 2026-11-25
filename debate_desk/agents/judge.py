"""Judge Agent: weighs both cases and issues BUY / HOLD / AVOID with the deciding reasons.

STRUCTURAL RULE: the Judge may only use cited facts. This is enforced three ways, none of which relies on
the model behaving:
  1. The Judge's input contains ONLY facts cited somewhere in the debate (it never receives the rest).
  2. The Judge gets no retrieval tool.
  3. Any fact ID the Judge outputs or mentions that was not cited in the debate is stripped and logged.
"""
from __future__ import annotations

import re
from dataclasses import asdict

from ..citations import clean_cites
from ..models import BEAR, BULL, Argument, AuditEntry, FactSheet, Judgement, Rebuttal, ScoreCard
from ..scoring import score_cases, verdict_from_margin
from .base import DebateAgent, fact_view

_ID = re.compile(r"\b[A-Z]{2,4}-\d{2,3}\b")
VERDICTS = {"BUY", "HOLD", "AVOID"}

INSTRUCTIONS = """You are the JUDGE of an investment committee debate. You decide BUY, HOLD or AVOID for the asset.
You are given ONLY the facts that the debaters cited (`cited_facts`), both cases, the rebuttals, and a
deterministic `scorecard` that weighs the cited evidence. You have no other information and must not use any.
Rules:
- Base your verdict ONLY on `cited_facts`. Cite the deciding facts by ID, and never mention an ID that is not in `cited_facts`.
- Judge the evidence, not the volume of argument. Consider which points survived rebuttal.
- You may depart from the scorecard's verdict, but then you must explain why using cited facts.
Reply as: {"verdict": "BUY|HOLD|AVOID", "confidence": 0.0-1.0, "rationale": "<3-5 sentences citing [IDs]>",
"deciding_facts": ["<fact id>"], "strongest_bull_index": <int>, "strongest_bear_index": <int>}"""


def short(text: str, n: int = 170) -> str:
    first = re.split(r";|\. ", text)[0].strip().rstrip(".")
    return first if len(first) <= n else first[: n - 1].rstrip() + "…"


def _strongest(card: dict, side: str) -> int:
    rows = [c for c in card["claim_scores"] if c["side"] == side]
    return max(rows, key=lambda c: c["effective"])["index"] if rows else -1


def judge_offline(payload: dict) -> dict:
    card, th = payload["scorecard"], payload["thresholds"]
    m = card["margin"]
    verdict = "BUY" if m >= th["buy"] else "AVOID" if m <= th["avoid"] else "HOLD"
    texts = {f["id"]: f["text"] for f in payload["cited_facts"]}
    top = lambda side, n: [k for k, _ in sorted(card["credited"][side].items(), key=lambda kv: -kv[1])[:n]]
    deciding = top("bull", 3) if verdict == "BUY" else top("bear", 3) if verdict == "AVOID" else top("bull", 2) + top("bear", 2)
    if verdict == "HOLD":
        conf = 0.5 + max(0.0, min(0.3, th["buy"] - abs(m)))
    else:
        conf = 0.5 + min(0.45, abs(m))

    s = [f"The Bull case scores {card['bull_total']:.2f} against the Bear case's {card['bear_total']:.2f} on cited evidence "
         f"(normalised margin {m:+.2f}; BUY needs ≥ {th['buy']:+.2f}, AVOID needs ≤ {th['avoid']:+.2f})."]
    if verdict == "BUY":
        s.append("Buy is justified by: " + "; ".join(f"{short(texts[f])} [{f}]" for f in deciding) + ".")
    elif verdict == "AVOID":
        s.append("Avoid is justified by: " + "; ".join(f"{short(texts[f])} [{f}]" for f in deciding) + ".")
    else:
        s.append("Neither side clears the decision margin. Strongest on the bull side: "
                 + "; ".join(f"{short(texts[f])} [{f}]" for f in top("bull", 2)) + ". Strongest on the bear side: "
                 + "; ".join(f"{short(texts[f])} [{f}]" for f in top("bear", 2)) + ".")
    for e in card["rebuttal_effects"]:
        who, tgt = e["rebutting_side"].capitalize(), e["target_side"].capitalize()
        if e["conceded"]:
            s.append(f"The {who} conceded the {tgt}'s strongest point.")
        elif e["effect"] > 0:
            s.append(f"The {who}'s rebuttal reduced the weight of the {tgt}'s strongest point by {round(100 * e['effect'] * payload['thresholds']['rebuttal_max_discount'])}%.")
    return {"verdict": verdict, "confidence": round(conf, 2), "rationale": " ".join(s), "deciding_facts": deciding,
            "strongest_bull_index": _strongest(card, "bull"), "strongest_bear_index": _strongest(card, "bear")}


class JudgeAgent(DebateAgent):
    role = "judge"
    retrieval = False            # the Judge gets no search tool
    instructions = INSTRUCTIONS

    async def run(self, sheet: FactSheet, args: dict[str, Argument], rebuttals: list[Rebuttal]) -> tuple[Judgement, ScoreCard]:
        cited: list[str] = []
        for a in args.values():
            cited += [f for c in a.claims for f in c.cites]
        for r in rebuttals:
            cited += r.cites
        cited = list(dict.fromkeys(cited))
        cited_set = set(cited)

        card = score_cases(sheet, args, rebuttals, self.cfg)
        scorecard_verdict = verdict_from_margin(card.margin, self.cfg)
        payload = {
            "asset": sheet.asset,
            "cited_facts": [fact_view(sheet, f) for f in cited],      # <- the ONLY facts the Judge ever sees
            "bull_case": [{"index": i, "text": c.text, "cites": c.cites} for i, c in enumerate(args[BULL].claims)],
            "bear_case": [{"index": i, "text": c.text, "cites": c.cites} for i, c in enumerate(args[BEAR].claims)],
            "rebuttals": [{"side": r.side, "target_side": r.target_side, "target_index": r.target_index,
                           "response": r.response, "cites": r.cites, "conceded": r.conceded} for r in rebuttals],
            "scorecard": asdict(card),
            "thresholds": {"buy": self.cfg.buy_margin, "avoid": self.cfg.avoid_margin,
                           "rebuttal_max_discount": self.cfg.rebuttal_max_discount},
        }
        raw = await self._call(payload, judge_offline)

        verdict = str(raw.get("verdict", "")).upper().strip()
        if verdict not in VERDICTS:
            self.audit.append(AuditEntry(self.role, "invalid_verdict", f"Judge returned {raw.get('verdict')!r}; scorecard verdict {scorecard_verdict} used."))
            verdict = scorecard_verdict

        deciding, rejected = clean_cites(raw.get("deciding_facts"), set(sheet.facts), allowed=cited_set)
        if rejected:
            self.audit.append(AuditEntry(self.role, "judge_used_uncited_fact", f"Stripped {rejected} from deciding facts: not cited in the debate."))
        if not deciding:
            ranked = sorted(card.credited[BULL if verdict == "BUY" else BEAR].items(), key=lambda kv: -kv[1])
            deciding = [k for k, _ in ranked[:3]] or cited[:3]
            self.audit.append(AuditEntry(self.role, "judge_no_valid_cites", "Judge named no valid deciding facts; top scorecard facts used."))

        rationale = str(raw.get("rationale", "")).strip()
        stray = sorted({i for i in _ID.findall(rationale) if i not in cited_set})
        if stray:
            for i in stray:
                rationale = re.sub(rf"\s*\[?{re.escape(i)}\]?", "", rationale)
            self.audit.append(AuditEntry(self.role, "judge_used_uncited_fact", f"Removed mentions of {stray} from the rationale: not cited in the debate."))
        if not rationale:
            rationale = judge_offline(payload)["rationale"]
            self.audit.append(AuditEntry(self.role, "empty_rationale", "Judge returned no rationale; scorecard rationale used."))

        try:
            conf = min(0.99, max(0.0, float(raw.get("confidence"))))
        except (TypeError, ValueError):
            conf = 0.5

        def idx(key: str, side: str, n: int) -> int:
            v = raw.get(key)
            return v if isinstance(v, int) and 0 <= v < n else _strongest(asdict(card), side)

        return Judgement(verdict=verdict, confidence=round(conf, 2), rationale=rationale, deciding_facts=deciding,
                         strongest_bull=idx("strongest_bull_index", BULL, len(args[BULL].claims)),
                         strongest_bear=idx("strongest_bear_index", BEAR, len(args[BEAR].claims)),
                         scorecard_verdict=scorecard_verdict, cited_fact_ids=cited), card
