"""Render the committee's output: Markdown report, JSON record, and a short console summary."""
from __future__ import annotations

import json
from dataclasses import asdict

from .models import BEAR, BULL
from .orchestrator import DebateState

BADGE = {"BUY": "BUY", "HOLD": "HOLD", "AVOID": "AVOID"}


def _tags(ids: list[str]) -> str:
    return " ".join(f"[{i}]" for i in ids)


def render_markdown(s: DebateState) -> str:
    sheet, j, card, r, f = s.sheet, s.judgement, s.scorecard, s.risk, s.final
    out: list[str] = [f"# Investment Committee Report: {s.asset}", ""]

    out += [f"## Recommendation: {BADGE[f.recommendation]}", ""]
    out.append(f"- **Judge's verdict:** {j.verdict} (confidence {j.confidence:.2f})")
    if f.veto_applied:
        out.append(f"- **Risk Agent:** VETO. The Judge's {j.verdict} was overridden and forced to HOLD.")
        out += [f"  - {x}" for x in f.veto_reasons]
    elif r.veto:
        out.append(f"- **Risk Agent:** veto condition met, but the veto only overrides a BUY, so the Judge's {j.verdict} stands.")
    else:
        out.append("- **Risk Agent:** no veto.")
    out += [f"- **Backend:** {s.backend}", ""]

    out += ["## Strongest argument on each side", ""]
    for side, idx, label in ((BULL, j.strongest_bull, "Bull (BUY)"), (BEAR, j.strongest_bear, "Bear (AVOID)")):
        rows = [c for c in card.claim_scores if c.side == side and c.index == idx]
        if rows:
            c = rows[0]
            out += [f"**{label}:** {c.text} {_tags(c.cites)}", "",
                    f"Evidence weight {c.base:.2f}; after the rebuttal {c.effective:.2f}.", ""]
        else:
            out += [f"**{label}:** no validly cited argument was presented.", ""]

    out += ["## Judge's deciding rationale", "", j.rationale, "", "Deciding facts:"]
    out += [f"- [{fid}] {sheet.facts[fid].text}" for fid in j.deciding_facts]
    if j.scorecard_verdict != j.verdict:
        out += ["", f"_Note: the deterministic scorecard alone gives {j.scorecard_verdict}; the Judge departed from it._"]
    out.append("")

    out += ["## Risk Agent review", "", f"Composite risk score: **{r.score}/100**", "", r.narrative, ""]
    if r.breaches:
        out += ["| Check | Observed | Limit | Status | Fact |", "|---|---|---|---|---|"]
        status = {"limit": "BREACHED", "flag": "HARD FLAG", "near": "near limit"}
        out += [f"| {b.label} | {b.observed} | {b.limit} | {status[b.kind]} | {_tags(b.cites)} |" for b in r.breaches]
        out.append("")

    out += ["## Scorecard (cited evidence only)", "",
            f"Bull {card.bull_total:.2f} vs Bear {card.bear_total:.2f}, normalised margin {card.margin:+.3f}", "",
            "| Side | # | Claim | Base | Rebuttal discount | Effective |", "|---|---|---|---|---|---|"]
    for c in card.claim_scores:
        out.append(f"| {c.side} | {c.index} | {c.text[:70]}… {_tags(c.cites)} | {c.base:.2f} | x{c.discount:.2f} | {c.effective:.2f} |")
    out.append("")

    out += ["## Debate transcript", "", "### Opening arguments"]
    for side in (BULL, BEAR):
        out += ["", f"**{side.capitalize()} Agent**"]
        out += [f"{i + 1}. {c.text} {_tags(c.cites)}" for i, c in enumerate(s.arguments[side].claims)] or ["_(no valid claims)_"]
    out += ["", "### Rebuttal round"]
    for rb in s.rebuttals:
        tag = "CONCEDES" if rb.conceded else "rebuts"
        out += ["", f"**{rb.side.capitalize()} Agent** {tag} the {rb.target_side}'s claim #{rb.target_index + 1}: {rb.response}"]
    out.append("")

    cited = j.cited_fact_ids
    out += ["## Evidence cited in the debate", "", "| ID | Source | Fact |", "|---|---|---|"]
    out += [f"| {fid} | {sheet.facts[fid].source} | {sheet.facts[fid].text} |" for fid in cited]
    out += ["", f"Facts in the document set the Judge never saw (not cited by either side): {len(sheet.facts) - len(cited)} of {len(sheet.facts)}.", ""]

    out += ["## Audit log", ""]
    out += [f"- `{a.agent}` {a.kind}: {a.detail}" for a in s.audit] or ["- No citation violations or fallbacks."]
    return "\n".join(out) + "\n"


def to_json(s: DebateState) -> str:
    return json.dumps(asdict(s), indent=2, default=str)


def console_summary(s: DebateState) -> str:
    f, j, r = s.final, s.judgement, s.risk
    bar = "=" * 72
    lines = [bar, f"  {s.asset}", bar, f"  RECOMMENDATION : {f.recommendation}"]
    lines.append(f"  Judge verdict  : {j.verdict} (confidence {j.confidence:.2f}); scorecard margin {s.scorecard.margin:+.2f}")
    lines.append(f"  Risk Agent     : score {r.score}/100, " + ("VETO" + (" -> forced HOLD" if f.veto_applied else " (not applicable to this verdict)") if r.veto else "no veto"))
    for x in f.veto_reasons:
        lines.append(f"                   - {x}")
    for side, idx in ((BULL, j.strongest_bull), (BEAR, j.strongest_bear)):
        rows = [c for c in s.scorecard.claim_scores if c.side == side and c.index == idx]
        if rows:
            lines.append(f"  Strongest {side:<5}  : {rows[0].text[:100]} {_tags(rows[0].cites)}")
    lines.append(f"  Audit entries  : {len(s.audit)}")
    lines.append(bar)
    return "\n".join(lines)
