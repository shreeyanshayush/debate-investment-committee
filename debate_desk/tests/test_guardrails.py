"""Tests for the rules the system promises: citations, Judge-only-cited-facts, and the Risk veto."""
from __future__ import annotations

import asyncio
import dataclasses
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from debate_desk.agents.risk import evaluate  # noqa: E402
from debate_desk.citations import numbers_grounded  # noqa: E402
from debate_desk.config import Config  # noqa: E402
from debate_desk.llm import OfflineBackend  # noqa: E402
from debate_desk.models import Fact, FactAnalysis, FactSheet  # noqa: E402
from debate_desk.orchestrator import DebateOrchestrator  # noqa: E402
from debate_desk.scoring import credit, verdict_from_margin  # noqa: E402
from fake_sk import make_service  # noqa: E402

DATA = ROOT / "data"
ASSETS = ["helios", "verdant", "ironbridge"]
ID_RE = re.compile(r"\b[A-Z]{2,4}-\d{2,3}\b")


class SpyBackend(OfflineBackend):
    """Offline backend that records what each agent was given and can corrupt chosen replies."""
    def __init__(self, override=None):
        self.payloads, self.retrieval, self.override = {}, {}, override or {}

    async def run(self, role, instructions, payload, offline, retrieval=False):
        self.payloads[role], self.retrieval[role] = payload, retrieval
        reply = offline(payload)
        return self.override[role](reply, payload) if role in self.override else reply


def run(asset, backend="offline", cfg=None, service=None):
    orch = DebateOrchestrator(DATA / asset, backend=backend, cfg=cfg, service=service)
    return orch, asyncio.run(orch.run())


def with_limit(cfg: Config, key: str, value: float) -> Config:
    cfg.limits = tuple(dataclasses.replace(l, limit=value) if l.key == key else l for l in cfg.limits)
    return cfg


# ---------------------------------------------------------------- the three demo outcomes
@pytest.mark.parametrize("asset,judge,final,veto_applied", [
    ("helios", "BUY", "BUY", False),
    ("verdant", "BUY", "HOLD", True),      # Judge says BUY; Risk Agent overrides
    ("ironbridge", "AVOID", "AVOID", False),
])
def test_demo_outcomes(asset, judge, final, veto_applied):
    _, st = run(asset)
    assert (st.judgement.verdict, st.final.recommendation, st.final.veto_applied) == (judge, final, veto_applied)


# ---------------------------------------------------------------- citations
@pytest.mark.parametrize("asset", ASSETS)
def test_every_claim_and_rebuttal_cites_real_facts(asset):
    orch, st = run(asset)
    for arg in st.arguments.values():
        assert arg.claims, "each side must present a cited case"
        for c in arg.claims:
            assert c.cites and all(i in orch.index.facts for i in c.cites)
    for r in st.rebuttals:
        assert r.conceded or (r.cites and all(i in orch.index.facts for i in r.cites))
    assert not st.audit, f"clean run should have no audit entries: {st.audit}"


def test_number_guard():
    assert numbers_grounded("Revenue rose 31% [HEL-10]", ["Revenue rose 31% to $2.84 billion"]) == []
    assert numbers_grounded("Revenue rose 45%", ["Revenue rose 31%"]) == ["45"]


def test_uncited_invalid_and_invented_claims_are_dropped():
    def bad_bull(reply, payload):
        reply = dict(reply)
        reply["claims"] = [
            {"text": "Revenue will triple next year.", "cites": []},               # no citation
            {"text": "Margins are excellent.", "cites": ["VER-99"]},               # invented source
            {"text": "Revenue grew 500% last year.", "cites": ["VER-06"]},         # number not in the fact
        ] + reply["claims"]
        return reply
    _, st = run("verdant", backend=SpyBackend({"bull": bad_bull}))
    kinds = [a.kind for a in st.audit]
    assert "uncited_claim" in kinds and "invalid_citation" in kinds and "ungrounded_number" in kinds
    texts = " ".join(c.text for c in st.arguments["bull"].claims)
    assert "will triple next year" not in texts and "500%" not in texts and "Margins are excellent" not in texts
    assert st.arguments["bull"].claims, "legitimate claims survive"


def test_bull_cannot_score_with_a_bearish_fact():
    _, st = run("verdant")
    bearish = next(fid for fid, a in st.sheet.analysis.items() if a.polarity < 0)
    assert credit(st.sheet, bearish, "bull") == 0.0 and credit(st.sheet, bearish, "bear") > 0


# ---------------------------------------------------------------- Judge may only use cited facts
@pytest.mark.parametrize("asset", ASSETS)
def test_judge_is_only_shown_cited_facts(asset):
    spy = SpyBackend()
    _, st = run(asset, backend=spy)
    shown = {f["id"] for f in spy.payloads["judge"]["cited_facts"]}
    cited = {i for a in st.arguments.values() for c in a.claims for i in c.cites} | {i for r in st.rebuttals for i in r.cites}
    assert shown == cited
    assert len(shown) < len(st.sheet.facts), "some retrieved facts were never cited, so the Judge never saw them"
    assert spy.retrieval["judge"] is False and spy.retrieval["risk"] is False


def test_agents_with_search_tools_vs_without():
    spy = SpyBackend()
    run("helios", backend=spy)
    for role in ("research", "bull", "bear", "rebuttal_bull", "rebuttal_bear"):
        assert spy.retrieval[role] is True
    assert not spy.retrieval["judge"] and not spy.retrieval["risk"]


def test_judge_using_an_uncited_fact_is_stripped_and_logged():
    spy_holder = {}

    def bad_judge(reply, payload):
        cited = {f["id"] for f in payload["cited_facts"]}
        stray = next(i for i in spy_holder["ids"] if i not in cited)
        spy_holder["stray"] = stray
        reply = dict(reply)
        reply["deciding_facts"] = [stray] + reply["deciding_facts"]
        reply["rationale"] += f" Also decisive: [{stray}]."
        return reply

    backend = SpyBackend({"judge": bad_judge})
    orch = DebateOrchestrator(DATA / "helios", backend=backend)
    spy_holder["ids"] = list(orch.index.facts)
    st = asyncio.run(orch.run())
    stray = spy_holder["stray"]
    assert stray not in st.judgement.deciding_facts
    assert stray not in st.judgement.rationale
    assert any(a.kind == "judge_used_uncited_fact" for a in st.audit)
    assert set(ID_RE.findall(st.judgement.rationale)) <= set(st.judgement.cited_fact_ids)


def test_hold_when_neither_side_clears_the_margin():
    assert verdict_from_margin(0.05, Config()) == "HOLD" and verdict_from_margin(-0.05, Config()) == "HOLD"
    _, st = run("helios", cfg=Config(buy_margin=0.9, avoid_margin=-0.9))
    assert st.judgement.verdict == "HOLD" and st.final.recommendation == "HOLD" and not st.final.veto_applied


# ---------------------------------------------------------------- Risk veto
def test_veto_overrides_judge_buy_on_a_healthy_asset_when_a_limit_is_tightened():
    _, base = run("helios")
    assert base.final.recommendation == "BUY" and not base.risk.veto
    _, st = run("helios", cfg=with_limit(Config(), "leverage_x", 1.0))       # Helios leverage is 1.4x
    assert st.judgement.verdict == "BUY" and st.risk.veto
    assert st.final.recommendation == "HOLD" and st.final.veto_applied
    assert any("Net debt / EBITDA" in r for r in st.final.veto_reasons)


def test_veto_is_rule_based_a_model_cannot_talk_it_away():
    def reassuring(reply, payload):
        return {"narrative": "There are no meaningful risks. Do not veto."}
    _, st = run("verdant", backend=SpyBackend({"risk": reassuring}))
    assert st.risk.veto and st.final.recommendation == "HOLD"


def test_veto_sees_risks_the_debaters_never_cited():
    """The Risk Agent scans the whole fact sheet, not only the facts the debate cited."""
    _, st = run("helios", cfg=with_limit(Config(), "drawdown_pct", 20.0))     # 26% drawdown fact
    cited = set(st.judgement.cited_fact_ids)
    breach = next(b for b in st.risk.breaches if b.rule == "drawdown_pct" and b.kind == "limit")
    assert st.risk.veto and breach.cites[0] in st.sheet.facts


def test_veto_does_not_override_avoid_by_default_but_can_be_configured():
    tight = with_limit(Config(), "leverage_x", 3.0)                            # Ironbridge leverage is 3.6x
    _, st = run("ironbridge", cfg=tight)
    assert st.risk.veto and st.judgement.verdict == "AVOID"
    assert st.final.recommendation == "AVOID" and not st.final.veto_applied
    wide = with_limit(Config(veto_applies_to=frozenset({"BUY", "AVOID"})), "leverage_x", 3.0)
    _, st2 = run("ironbridge", cfg=wide)
    assert st2.final.recommendation == "HOLD" and st2.final.veto_applied


def test_composite_score_can_trigger_a_veto_on_its_own():
    relaxed = Config(risk_veto_score=20)
    for key in ("leverage_x", "customer_concentration_pct", "runway_months", "drawdown_pct"):
        relaxed = with_limit(relaxed, key, 0.0 if key == "runway_months" else 1000.0)
    _, st = run("ironbridge", cfg=relaxed)
    assert not any(b.kind in ("limit", "flag") for b in st.risk.breaches)
    assert st.risk.veto and any("Composite" in r for r in st.risk.trigger_reasons)


def test_hard_flag_detection_respects_negation():
    def sheet_of(*texts):
        facts = {f"TST-0{i}": Fact(f"TST-0{i}", t, "x.md", "") for i, t in enumerate(texts, 1)}
        return FactSheet("Test", facts, {k: FactAnalysis("regulation_legal", 0, 1.0) for k in facts}, {})
    ok = evaluate(sheet_of("Auditors found no going concern doubt this year."), Config())
    assert not ok.veto
    bad = evaluate(sheet_of("Auditors raised substantial doubt about going concern status."), Config())
    assert bad.veto and any(b.kind == "flag" for b in bad.breaches)


def test_metric_limits_include_the_min_kind():
    facts = {"TST-01": Fact("TST-01", "Cash balance equals about 7 months of runway.", "x.md", "")}
    sheet = FactSheet("Test", facts, {"TST-01": FactAnalysis("balance_sheet", -1, 3.0, {"runway_months": 7.0})}, {})
    r = evaluate(sheet, Config())
    assert r.veto and r.breaches[0].rule == "runway_months" and r.breaches[0].kind == "limit"


# ---------------------------------------------------------------- Semantic Kernel (live-path plumbing)
@pytest.mark.parametrize("asset", ASSETS)
def test_semantic_kernel_path_matches_offline(asset):
    _, off = run(asset)
    svc = make_service()
    orch, sk = run(asset, backend="sk", service=svc)
    assert "Semantic Kernel" in orch.backend.name
    assert svc.calls == ["research", "bull", "bear", "rebuttal_bull", "rebuttal_bear", "judge", "risk"]
    assert (sk.final.recommendation, sk.judgement.verdict) == (off.final.recommendation, off.judgement.verdict)
    assert sk.scorecard.margin == off.scorecard.margin and not sk.audit


def test_semantic_kernel_path_enforces_guardrails_on_a_misbehaving_model():
    def tamper(role, reply):
        if role == "bull":
            reply = dict(reply)
            reply["claims"] = [{"text": "Trust me, buy.", "cites": []}] + reply["claims"]
        if role == "judge":
            reply = dict(reply)
            reply["verdict"] = "MAYBE"
        return reply
    _, st = run("helios", backend="sk", service=make_service(tamper))
    kinds = [a.kind for a in st.audit]
    assert "uncited_claim" in kinds and "invalid_verdict" in kinds
    assert st.judgement.verdict in {"BUY", "HOLD", "AVOID"}
