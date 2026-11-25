"""Debate-pattern orchestration (proponent vs opponent + judge, with an independent risk veto).

    Research ──► Bull ┐                     ┌─► Judge ┐
                      ├─► Rebuttal round ──┤         ├─► resolve (Risk veto may force HOLD)
                Bear ─┘                     └─► Risk ─┘

`DebateState` is the shared memory (blackboard) every stage reads from and writes to.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path

from .agents.base import Context
from .agents.bear import BearAgent
from .agents.bull import BullAgent
from .agents.judge import JudgeAgent
from .agents.rebuttal import RebuttalRound
from .agents.research import ResearchAgent
from .agents.risk import RiskAgent
from .config import Config
from .llm import Backend, OfflineBackend, SKBackend, model_configured
from .models import (BEAR, BULL, Argument, AuditEntry, FactSheet, FinalDecision, Judgement, Rebuttal,
                     RiskAssessment, ScoreCard)
from .rag import DocumentIndex


@dataclass
class DebateState:
    asset: str
    backend: str
    sheet: FactSheet | None = None
    arguments: dict[str, Argument] = field(default_factory=dict)
    rebuttals: list[Rebuttal] = field(default_factory=list)
    judgement: Judgement | None = None
    scorecard: ScoreCard | None = None
    risk: RiskAssessment | None = None
    final: FinalDecision | None = None
    audit: list[AuditEntry] = field(default_factory=list)
    transcript: list[dict] = field(default_factory=list)


def make_backend(kind: str, index: DocumentIndex, service=None) -> Backend:
    if kind == "offline":
        return OfflineBackend()
    if kind == "sk" or (kind == "auto" and (service is not None or model_configured())):
        return SKBackend(index, service=service)
    return OfflineBackend()


class DebateOrchestrator:
    def __init__(self, asset_dir: str | Path, backend: str | Backend = "auto", cfg: Config | None = None, service=None):
        self.cfg = cfg or Config()
        self.index = DocumentIndex(asset_dir)
        self.backend: Backend = make_backend(backend, self.index, service) if isinstance(backend, str) else backend

    async def run(self) -> DebateState:
        cfg, be = self.cfg, self.backend
        st = DebateState(asset=self.index.asset_name, backend=be.name)
        ctx = Context(self.index, cfg, st.audit)
        log = lambda stage, agent, content: st.transcript.append({"stage": stage, "agent": agent, "content": content})

        # 1. Research: assemble the facts (RAG) -> shared fact sheet
        st.sheet = await ResearchAgent(be, ctx).run()
        log("research", "Research Agent", f"Assembled {len(st.sheet.facts)} facts from {len(self.index)} in the document set.")

        # 2. Opening arguments (independent, concurrent)
        bull, bear = await asyncio.gather(BullAgent(be, ctx).run(st.sheet), BearAgent(be, ctx).run(st.sheet))
        st.arguments = {BULL: bull, BEAR: bear}
        log("opening", "Bull Agent", [c.__dict__ for c in bull.claims])
        log("opening", "Bear Agent", [c.__dict__ for c in bear.claims])

        # 3. Rebuttal round: each side answers the other's strongest point once
        st.rebuttals = await RebuttalRound(be, ctx).run(st.sheet, st.arguments)
        for r in st.rebuttals:
            log("rebuttal", f"{r.side.capitalize()} Agent", r.__dict__)

        # 4. Judge (cited facts only) and Risk (whole fact sheet) run independently
        (st.judgement, st.scorecard), st.risk = await asyncio.gather(
            JudgeAgent(be, ctx).run(st.sheet, st.arguments, st.rebuttals),
            RiskAgent(be, ctx).run(st.sheet),
        )
        log("judge", "Judge Agent", st.judgement.__dict__)
        log("risk", "Risk Agent", {"score": st.risk.score, "veto": st.risk.veto, "reasons": st.risk.trigger_reasons})

        # 5. Resolve: the Risk Agent's veto overrides the Judge
        applied = st.risk.veto and st.judgement.verdict in cfg.veto_applies_to
        st.final = FinalDecision(asset=st.asset, recommendation="HOLD" if applied else st.judgement.verdict,
                                 judge_verdict=st.judgement.verdict, veto_applied=applied,
                                 veto_reasons=st.risk.trigger_reasons if applied else [])
        log("decision", "Committee", st.final.__dict__)
        return st
