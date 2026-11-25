"""Typed data passed between agents (the shared state of the debate)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .config import THEME_WEIGHTS

BULL, BEAR = "bull", "bear"
SIGN = {BULL: 1, BEAR: -1}
OPPONENT = {BULL: BEAR, BEAR: BULL}


@dataclass(frozen=True)
class Fact:
    """One citable unit of evidence from the document set."""
    id: str
    text: str
    source: str
    section: str


@dataclass
class FactAnalysis:
    theme: str
    polarity: int            # +1 bullish, -1 bearish, 0 neutral/mixed
    strength: float          # 0.5 .. 3.0
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass
class FactSheet:
    """Output of the Research Agent."""
    asset: str
    facts: dict[str, Fact]
    analysis: dict[str, FactAnalysis]
    retrieved_by: dict[str, list[str]]

    def weight(self, fid: str) -> float:
        a = self.analysis[fid]
        return a.strength * THEME_WEIGHTS.get(a.theme, 0.7)


@dataclass
class Claim:
    text: str
    cites: list[str]


@dataclass
class Argument:
    side: str
    claims: list[Claim]


@dataclass
class Rebuttal:
    side: str                # who is rebutting
    target_side: str
    target_index: int        # index of the opponent claim being answered
    response: str
    cites: list[str]
    conceded: bool = False


@dataclass
class AuditEntry:
    agent: str
    kind: str
    detail: str


@dataclass
class ClaimScore:
    side: str
    index: int
    text: str
    cites: list[str]
    base: float
    discount: float
    effective: float


@dataclass
class ScoreCard:
    bull_total: float
    bear_total: float
    margin: float
    claim_scores: list[ClaimScore]
    credited: dict[str, dict[str, float]]      # side -> fact id -> credit
    rebuttal_effects: list[dict[str, Any]]


@dataclass
class Judgement:
    verdict: str
    confidence: float
    rationale: str
    deciding_facts: list[str]
    strongest_bull: int      # index into bull claims
    strongest_bear: int
    scorecard_verdict: str   # what the deterministic scorecard alone says
    cited_fact_ids: list[str]


@dataclass
class Breach:
    rule: str
    label: str
    kind: str                # "limit" | "near" | "flag"
    observed: str
    limit: str
    cites: list[str]


@dataclass
class RiskAssessment:
    score: int
    breaches: list[Breach]
    veto: bool
    trigger_reasons: list[str]
    narrative: str


@dataclass
class FinalDecision:
    asset: str
    recommendation: str
    judge_verdict: str
    veto_applied: bool
    veto_reasons: list[str]
